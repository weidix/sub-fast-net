use std::{
    fs::{self, OpenOptions},
    io::Write,
    marker::PhantomData,
    path::Path,
    sync::{
        Arc,
        mpsc::{Receiver, sync_channel},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use burn::{
    module::{AutodiffModule, ModuleVisitor, Param},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Bool, BoolStore, DType, Tensor, backend::AutodiffBackend},
};
use serde::Serialize;

use crate::{
    checkpoint::{
        CheckpointMeta, load_optimizer_record, load_scheduler_state, model_artifact_size_bytes,
        save_checkpoint, save_optimizer_record,
    },
    config::{MixedPrecision, ProfilingAblation, TrainConfig},
    dataset::SubtitleDataset,
    loss::{
        LossComponentSelection, LossTensorCache,
        compute_tensor_loss_breakdown_cached_with_selection,
    },
    model::{SubFastNet, serialized_size_bytes_estimate},
    preprocess::TrainingBatch,
    preprocess::{collate_batch_with_config, preprocess_sample},
    validate::{validate_model, write_error_reports},
};

const FP16_INITIAL_LOSS_SCALE: f32 = 1024.0;
const FP16_MIN_LOSS_SCALE: f32 = 1.0;
const FP16_MAX_LOSS_SCALE: f32 = 65536.0;
const FP16_LOSS_SCALE_GROWTH_INTERVAL: usize = 2000;

#[derive(Debug, Clone, Serialize)]
pub struct TrainStepMetrics {
    pub epoch: usize,
    pub step: usize,
    pub epoch_batch: usize,
    pub epoch_batches: usize,
    pub epoch_samples_processed: usize,
    pub epoch_samples_total: usize,
    pub learning_rate: f32,
    pub total_loss: f32,
    pub region_loss: f32,
    pub kernel_loss: f32,
    pub bbox_loss: f32,
    pub samples_per_second: f32,
    pub batch_time: f32,
    pub data_time: f32,
    pub ignored_area_ratio: f32,
    pub positive_region_ratio: f32,
    pub positive_kernel_ratio: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingSummary {
    pub best_f1: f32,
    pub final_epoch: usize,
    pub final_step: usize,
    pub model_size_bytes_estimate: usize,
    pub final_model_artifact_size_bytes: Option<u64>,
    pub model_size_target_min_bytes: u64,
    pub model_size_target_max_bytes: u64,
    pub final_model_size_within_target: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TrainProfilingStep {
    pub record_type: &'static str,
    pub epoch: usize,
    pub step: usize,
    pub epoch_batch: usize,
    pub batch_wall_time: f32,
    pub dataloader_data_time: f32,
    pub dataloader_wait_time: f32,
    pub h2d_copy_cpu_time: f32,
    pub h2d_image_tensor_cpu_time: f32,
    pub forward_cpu_time: f32,
    pub target_tensor_cpu_time: f32,
    pub loss_compute_cpu_time: f32,
    pub backward_cpu_time: f32,
    pub optimizer_step_cpu_time: f32,
    pub h2d_copy_gpu_time: f32,
    pub h2d_image_tensor_gpu_time: f32,
    pub h2d_target_tensor_gpu_time: f32,
    pub forward_gpu_time: f32,
    pub loss_compute_gpu_time: f32,
    pub backward_gpu_time: f32,
    pub optimizer_step_gpu_time: f32,
    pub target_gt_text_requires_grad: bool,
    pub target_gt_kernel_requires_grad: bool,
    pub target_training_mask_requires_grad: bool,
    pub backward_call_count: usize,
    pub profiling_ablation: ProfilingAblation,
    pub loss_to_cpu_breakdown_time: f32,
    pub metrics_write_time: f32,
    pub backend_sync_time: f32,
    pub gpu_timing_mode: &'static str,
    pub batch_time: f32,
    pub data_time: f32,
    pub wait_time: f32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TrainProfilingAverage {
    pub record_type: &'static str,
    pub skipped_cold_steps: usize,
    pub warm_steps: usize,
    pub batch_wall_time: f32,
    pub dataloader_data_time: f32,
    pub dataloader_wait_time: f32,
    pub h2d_copy_cpu_time: f32,
    pub h2d_image_tensor_cpu_time: f32,
    pub forward_cpu_time: f32,
    pub target_tensor_cpu_time: f32,
    pub loss_compute_cpu_time: f32,
    pub backward_cpu_time: f32,
    pub optimizer_step_cpu_time: f32,
    pub h2d_copy_gpu_time: f32,
    pub h2d_image_tensor_gpu_time: f32,
    pub h2d_target_tensor_gpu_time: f32,
    pub forward_gpu_time: f32,
    pub loss_compute_gpu_time: f32,
    pub backward_gpu_time: f32,
    pub optimizer_step_gpu_time: f32,
    pub target_gt_text_requires_grad: bool,
    pub target_gt_kernel_requires_grad: bool,
    pub target_training_mask_requires_grad: bool,
    pub max_backward_call_count: usize,
    pub profiling_ablation: ProfilingAblation,
    pub loss_to_cpu_breakdown_time: f32,
    pub metrics_write_time: f32,
    pub backend_sync_time: f32,
    pub gpu_timing_mode: &'static str,
    pub batch_time: f32,
    pub data_time: f32,
    pub wait_time: f32,
}

#[derive(Debug, Clone, Default)]
struct TrainProfiler {
    steps: Vec<TrainProfilingStep>,
}

enum TrainProfileDuration {}

trait TrainProfileTiming {
    fn mode(&self) -> &'static str;
    fn profile<T: Send + 'static>(
        &self,
        enabled: bool,
        name: &'static str,
        func: impl FnOnce() -> T + Send,
    ) -> Result<(T, Option<TrainProfileDuration>)>;
}

#[derive(Clone, Copy)]
struct WallClockProfiler;

impl TrainProfileTiming for WallClockProfiler {
    fn mode(&self) -> &'static str {
        "wall_time"
    }

    fn profile<T: Send + 'static>(
        &self,
        _enabled: bool,
        _name: &'static str,
        func: impl FnOnce() -> T + Send,
    ) -> Result<(T, Option<TrainProfileDuration>)> {
        Ok((func(), None))
    }
}

fn profile_section<T: Send + 'static>(
    timing: impl TrainProfileTiming,
    enabled: bool,
    name: &'static str,
    func: impl FnOnce() -> T + Send,
) -> Result<(T, f32, Option<TrainProfileDuration>)> {
    let start = Instant::now();
    let (output, duration) = timing.profile(enabled, name, func)?;
    let cpu_time = start.elapsed().as_secs_f32();
    Ok((output, cpu_time, duration))
}

fn resolve_profile_duration(_duration: Option<TrainProfileDuration>) -> Result<f32> {
    Ok(0.0)
}

pub fn train(config: &TrainConfig) -> Result<TrainingSummary> {
    crate::backend::train_with_configured_backend(config)
}

pub fn train_backend<B: burn::tensor::backend::AutodiffBackend>(
    config: &TrainConfig,
) -> Result<TrainingSummary> {
    config.save_snapshot()?;
    let device = B::Device::default();
    let profile_timing = WallClockProfiler;
    B::seed(&device, config.seed);
    let train_dataset = Arc::new(SubtitleDataset::from_train_config(config)?);
    let val_dataset = SubtitleDataset::from_val_config(config)?;
    let mut model = SubFastNet::<B>::new(config.model_variant, &device);
    let mut optimizer = AdamConfig::new().init::<B, SubFastNet<B>>();
    let mut loss_tensor_cache = LossTensorCache::<B>::new(&device);
    let mut loss_scaler = Fp16LossScaler::new(config.mixed_precision);
    let mut best_f1 = 0.0;
    let mut step = 0;
    let mut start_epoch = 1;
    if !config.resume.trim().is_empty() {
        let (resume_model, meta) =
            crate::checkpoint::load_checkpoint_model::<B>(&config.resume, &device)?;
        model = resume_model;
        optimizer = load_optimizer_record::<_, B>(&config.resume, optimizer)?;
        best_f1 = meta.best_f1;
        step = meta.step;
        start_epoch = load_scheduler_state(&config.resume)?
            .map(|state| state.epoch.saturating_add(1))
            .unwrap_or_else(|| meta.epoch.saturating_add(1));
    }
    fs::create_dir_all(&config.output_dir)?;
    let metrics_path = Path::new(&config.output_dir).join("metrics.jsonl");
    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(metrics_path)?;
    let mut profile_file = if config.profiling_enabled {
        let profile_path = Path::new(&config.output_dir).join("training_profile.jsonl");
        Some(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(profile_path)?,
        )
    } else {
        None
    };
    let mut profiler = TrainProfiler::default();
    let mut tui = crate::tui::BurnTui::new(config);

    for epoch in start_epoch..=config.epochs {
        let learning_rate = scheduled_learning_rate(config, epoch);
        let epoch_batches = train_dataset.len().div_ceil(config.batch_size.max(1));
        let (batch_receiver, prefetch_worker) =
            spawn_epoch_prefetch(Arc::clone(&train_dataset), config.clone());
        let mut prefetch_error = None;
        loop {
            let wait_start = Instant::now();
            let prepared = match next_prefetched_batch(&batch_receiver, &prefetch_worker) {
                Ok(Some(prepared)) => prepared,
                Ok(None) => break,
                Err(err) => {
                    prefetch_error = Some(err);
                    break;
                }
            };
            let wait_time = wait_start.elapsed();
            let batch_start = Instant::now();
            let batch_size = prepared.batch.imgs.len();
            let batch = prepared.batch;
            let mut profile = TrainProfilingStep::new(
                epoch,
                step + 1,
                prepared.epoch_batch,
                prepared.data_time,
                wait_time,
                gpu_timing_mode(config, profile_timing),
                config.profiling_ablation,
            );

            let (images, image_cpu_time, image_gpu_time) = profile_section(
                profile_timing,
                config.profiling_enabled,
                "h2d_images",
                || batch.image_tensor::<B>(&device),
            )?;
            if matches!(
                config.profiling_ablation,
                ProfilingAblation::ForwardOnly | ProfilingAblation::ForwardLossOnly
            ) {
                let h2d_image_gpu_time = resolve_profile_duration(image_gpu_time)?;
                let valid_model = model.clone().valid();
                let (output, forward_cpu_time, forward_gpu_time) =
                    profile_section(profile_timing, config.profiling_enabled, "forward", || {
                        valid_model.forward(images.inner())
                    })?;
                let mut target_cpu_time = 0.0;
                let mut target_gpu_time = None;
                let mut loss_cpu_time = 0.0;
                let mut loss_gpu_time = None;
                if config.profiling_ablation == ProfilingAblation::ForwardLossOnly {
                    let target_device = output.text_region_logits.device();
                    let (targets, cpu_time, gpu_time) = profile_section(
                        profile_timing,
                        config.profiling_enabled,
                        "h2d_targets",
                        || {
                            let gt_text = batch.text_tensor::<B::InnerBackend>(
                                output.height,
                                output.width,
                                &target_device,
                            );
                            let gt_kernel = batch.kernel_tensor::<B::InnerBackend>(
                                output.height,
                                output.width,
                                &target_device,
                            );
                            let training_mask = batch.training_mask_tensor::<B::InnerBackend>(
                                output.height,
                                output.width,
                                &target_device,
                            );
                            (gt_text, gt_kernel, training_mask)
                        },
                    )?;
                    target_cpu_time = cpu_time;
                    target_gpu_time = gpu_time;
                    let (gt_text, gt_kernel, training_mask) = targets;
                    profile.target_gt_text_requires_grad = gt_text.is_require_grad();
                    profile.target_gt_kernel_requires_grad = gt_kernel.is_require_grad();
                    profile.target_training_mask_requires_grad = training_mask.is_require_grad();
                    let mut inner_loss_tensor_cache =
                        LossTensorCache::<B::InnerBackend>::new(&target_device);
                    let (_loss, cpu_time, gpu_time) = profile_section(
                        profile_timing,
                        config.profiling_enabled,
                        "loss_compute",
                        || {
                            compute_tensor_loss_breakdown_cached_with_selection(
                                &output,
                                gt_text,
                                gt_kernel,
                                training_mask,
                                &mut inner_loss_tensor_cache,
                                LossComponentSelection::ALL,
                            )
                        },
                    )?;
                    loss_cpu_time = cpu_time;
                    loss_gpu_time = gpu_time;
                }

                let batch_time = batch_start.elapsed().as_secs_f32();
                let h2d_target_gpu_time = resolve_profile_duration(target_gpu_time)?;
                profile.loss_compute_gpu_time = resolve_profile_duration(loss_gpu_time)?;
                profile.forward_gpu_time = resolve_profile_duration(forward_gpu_time)?;
                profile.h2d_image_tensor_gpu_time = h2d_image_gpu_time;
                profile.h2d_target_tensor_gpu_time = h2d_target_gpu_time;
                profile.h2d_copy_gpu_time = h2d_image_gpu_time + h2d_target_gpu_time;
                profile.h2d_copy_cpu_time = image_cpu_time + target_cpu_time;
                profile.h2d_image_tensor_cpu_time = image_cpu_time;
                profile.forward_cpu_time = forward_cpu_time;
                profile.target_tensor_cpu_time = target_cpu_time;
                profile.loss_compute_cpu_time = loss_cpu_time;
                step += 1;
                profile.batch_wall_time = batch_time;
                profile.batch_time = batch_time;
                if let Some(profile_file) = &mut profile_file {
                    writeln!(profile_file, "{}", serde_json::to_string(&profile)?)?;
                    profiler.push(profile);
                }
                sync_training_device::<B>(&device, "after train step")?;
                continue;
            }
            let (output, forward_cpu_time, forward_gpu_time) = profile_section(
                profile_timing,
                config.profiling_enabled,
                "forward",
                || match config.profiling_ablation {
                    ProfilingAblation::HeadOnlyBackward => model.forward_head_only_backward(images),
                    _ => model.forward(images),
                },
            )?;
            let needs_loss = config.profiling_ablation.needs_loss();
            let needs_backward = config.profiling_ablation.needs_backward();
            let needs_optimizer = config.profiling_ablation.needs_optimizer();
            let mut loss = None;
            let mut target_cpu_time = 0.0;
            let mut target_gpu_time = None;
            let mut loss_cpu_time = 0.0;
            let mut loss_gpu_time = None;
            let mut backward_cpu_time = 0.0;
            let mut backward_gpu_time = None;
            let mut optimizer_cpu_time = 0.0;
            let mut optimizer_gpu_time = None;
            if needs_loss {
                let (targets, cpu_time, gpu_time) = profile_section(
                    profile_timing,
                    config.profiling_enabled,
                    "h2d_targets",
                    || {
                        let gt_text = batch.text_tensor::<B>(output.height, output.width, &device);
                        let gt_kernel =
                            batch.kernel_tensor::<B>(output.height, output.width, &device);
                        let training_mask =
                            batch.training_mask_tensor::<B>(output.height, output.width, &device);
                        (gt_text, gt_kernel, training_mask)
                    },
                )?;
                target_cpu_time = cpu_time;
                target_gpu_time = gpu_time;
                let (gt_text, gt_kernel, training_mask) = targets;
                profile.target_gt_text_requires_grad = gt_text.is_require_grad();
                profile.target_gt_kernel_requires_grad = gt_kernel.is_require_grad();
                profile.target_training_mask_requires_grad = training_mask.is_require_grad();
                let selection = loss_component_selection(config.profiling_ablation);
                let (computed_loss, cpu_time, gpu_time) = profile_section(
                    profile_timing,
                    config.profiling_enabled,
                    "loss_compute",
                    || {
                        compute_tensor_loss_breakdown_cached_with_selection(
                            &output,
                            gt_text,
                            gt_kernel,
                            training_mask,
                            &mut loss_tensor_cache,
                            selection,
                        )
                    },
                )?;
                loss_cpu_time = cpu_time;
                loss_gpu_time = gpu_time;
                loss = Some(computed_loss);
            }
            if needs_backward {
                if let Some(loss) = loss.as_ref()
                    && !loss.total_loss_is_finite()
                {
                    loss_scaler.backoff();
                    println!(
                        "mixed_precision_skip_update step={} reason=non_finite_loss loss_scale={:.1}",
                        step + 1,
                        loss_scaler.scale()
                    );
                    sync_training_device::<B>(&device, "after non-finite loss skip")?;
                    continue;
                }
                let (grads, cpu_time, gpu_time) =
                    profile_section(profile_timing, config.profiling_enabled, "backward", || {
                        profile_backward_loss(config.profiling_ablation, &output, loss.as_ref())
                            .map(|backward_loss| {
                                let backward_loss = loss_scaler.scale_loss(backward_loss);
                                GradientsParams::from_grads(backward_loss.backward(), &model)
                            })
                    })?;
                backward_cpu_time = cpu_time;
                backward_gpu_time = gpu_time;
                profile.backward_call_count = 1;
                let grads = grads.context("profiling ablation did not produce a backward loss")?;
                let (grads, grads_finite) = loss_scaler.unscale_and_check::<B, _>(grads, &model);
                if !grads_finite {
                    loss_scaler.backoff();
                    println!(
                        "mixed_precision_skip_update step={} reason=non_finite_grad loss_scale={:.1}",
                        step + 1,
                        loss_scaler.scale()
                    );
                    sync_training_device::<B>(&device, "after non-finite grad skip")?;
                    continue;
                }
                if needs_optimizer {
                    let (updated_model, cpu_time, gpu_time) = profile_section(
                        profile_timing,
                        config.profiling_enabled,
                        "optimizer_step",
                        || optimizer.step(learning_rate as f64, model, grads),
                    )?;
                    model = updated_model;
                    optimizer_cpu_time = cpu_time;
                    optimizer_gpu_time = gpu_time;
                    loss_scaler.update_after_success();
                }
            }

            let batch_time = batch_start.elapsed().as_secs_f32();

            let h2d_image_gpu_time = resolve_profile_duration(image_gpu_time)?;
            let h2d_target_gpu_time = resolve_profile_duration(target_gpu_time)?;
            profile.optimizer_step_gpu_time = resolve_profile_duration(optimizer_gpu_time)?;
            profile.backward_gpu_time = resolve_profile_duration(backward_gpu_time)?;
            profile.loss_compute_gpu_time = resolve_profile_duration(loss_gpu_time)?;
            profile.forward_gpu_time = resolve_profile_duration(forward_gpu_time)?;
            profile.h2d_image_tensor_gpu_time = h2d_image_gpu_time;
            profile.h2d_target_tensor_gpu_time = h2d_target_gpu_time;
            profile.h2d_copy_gpu_time = h2d_image_gpu_time + h2d_target_gpu_time;
            profile.h2d_copy_cpu_time = image_cpu_time + target_cpu_time;
            profile.h2d_image_tensor_cpu_time = image_cpu_time;
            profile.forward_cpu_time = forward_cpu_time;
            profile.target_tensor_cpu_time = target_cpu_time;
            profile.loss_compute_cpu_time = loss_cpu_time;
            profile.backward_cpu_time = backward_cpu_time;
            profile.optimizer_step_cpu_time = optimizer_cpu_time;

            step += 1;
            if let Some(loss) = loss {
                let timed_start = Instant::now();
                let loss = loss.to_cpu_loss_breakdown();
                profile.loss_to_cpu_breakdown_time = timed_start.elapsed().as_secs_f32();
                let metrics = TrainStepMetrics {
                    epoch,
                    step,
                    epoch_batch: prepared.epoch_batch,
                    epoch_batches,
                    epoch_samples_processed: prepared.samples_processed,
                    epoch_samples_total: prepared.samples_total,
                    learning_rate,
                    total_loss: loss.total_loss,
                    region_loss: loss.region_bce_loss + loss.region_dice_loss,
                    kernel_loss: loss.kernel_bce_loss + loss.kernel_dice_loss,
                    bbox_loss: loss.bbox_loss,
                    samples_per_second: batch_size as f32 / batch_time.max(1e-6),
                    batch_time,
                    data_time: prepared.data_time,
                    ignored_area_ratio: loss.ignored_area_ratio,
                    positive_region_ratio: loss.positive_region_ratio,
                    positive_kernel_ratio: loss.positive_kernel_ratio,
                };
                tui.update_train(&metrics);
                if step % config.log_interval.max(1) == 0 {
                    print_training_status(config, &metrics, tui.is_active());
                }
                let timed_start = Instant::now();
                writeln!(metrics_file, "{}", serde_json::to_string(&metrics)?)?;
                profile.metrics_write_time = timed_start.elapsed().as_secs_f32();
            }
            profile.batch_wall_time = batch_time;
            profile.batch_time = batch_time;
            if let Some(profile_file) = &mut profile_file {
                writeln!(profile_file, "{}", serde_json::to_string(&profile)?)?;
                profiler.push(profile);
            }
            sync_training_device::<B>(&device, "after train step")?;
        }
        drop(batch_receiver);
        let worker_result = join_prefetch_worker(prefetch_worker);
        if let Some(err) = prefetch_error {
            let _ = worker_result;
            return Err(err);
        }
        worker_result?;
        sync_training_device::<B>(&device, "after train epoch")?;
        if epoch % config.validation_interval.max(1) == 0 {
            let validation = validate_model(config, &val_dataset, &model.clone().valid())?;
            sync_training_device::<B>(&device, "after validation")?;
            writeln!(
                metrics_file,
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "record_type": "validation",
                    "epoch": epoch,
                    "step": step,
                    "metrics": validation,
                }))?
            )?;
            tui.update_valid(epoch, step, &validation);
            print_validation_status(config, epoch, step, &validation, tui.is_active());
            if validation.f1 >= best_f1 {
                best_f1 = validation.f1;
                save_training_checkpoint(
                    Path::new(&config.output_dir).join("best"),
                    model.clone().valid(),
                    &optimizer,
                    &CheckpointMeta {
                        epoch,
                        step,
                        best_f1,
                        config: config.clone(),
                        learning_rate,
                        scheduler_epoch: epoch,
                    },
                )?;
                println!("checkpoint=best epoch={epoch} step={step}");
            }
        }
        if epoch % config.checkpoint_interval.max(1) == 0 {
            save_training_checkpoint(
                Path::new(&config.output_dir)
                    .join("checkpoints")
                    .join(format!("epoch_{epoch:04}")),
                model.clone().valid(),
                &optimizer,
                &CheckpointMeta {
                    epoch,
                    step,
                    best_f1,
                    config: config.clone(),
                    learning_rate,
                    scheduler_epoch: epoch,
                },
            )?;
            println!("checkpoint=periodic epoch={epoch} step={step}");
        }
    }
    sync_training_device::<B>(&device, "before final checkpoint")?;
    tui.finish();
    save_training_checkpoint(
        Path::new(&config.output_dir).join("final"),
        model.clone().valid(),
        &optimizer,
        &CheckpointMeta {
            epoch: config.epochs,
            step,
            best_f1,
            config: config.clone(),
            learning_rate: scheduled_learning_rate(config, config.epochs),
            scheduler_epoch: config.epochs,
        },
    )?;
    sync_training_device::<B>(&device, "after final checkpoint")?;
    write_error_reports(config, &val_dataset, &model.clone().valid())?;
    sync_training_device::<B>(&device, "after error reports")?;
    let final_model_artifact_size_bytes =
        model_artifact_size_bytes(Path::new(&config.output_dir).join("final"))?;
    let model_size_target_min_bytes = 1_000_000;
    let model_size_target_max_bytes = 4_000_000;
    let final_model_size_within_target = final_model_artifact_size_bytes
        .map(|size| size >= model_size_target_min_bytes && size <= model_size_target_max_bytes);
    let summary = TrainingSummary {
        best_f1,
        final_epoch: config.epochs,
        final_step: step,
        model_size_bytes_estimate: serialized_size_bytes_estimate(config.model_variant),
        final_model_artifact_size_bytes,
        model_size_target_min_bytes,
        model_size_target_max_bytes,
        final_model_size_within_target,
    };
    fs::write(
        Path::new(&config.output_dir).join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    if config.profiling_enabled {
        let average = profiler.warm_average();
        fs::write(
            Path::new(&config.output_dir).join("training_profile_summary.json"),
            serde_json::to_string_pretty(&average)?,
        )?;
        print_profile_average(&average);
    }
    Ok(summary)
}

fn sync_training_device<B: AutodiffBackend>(device: &B::Device, stage: &'static str) -> Result<()> {
    B::sync(device).with_context(|| format!("failed to synchronize training device {stage}"))?;
    Ok(())
}

#[derive(Debug, Clone)]
struct Fp16LossScaler {
    enabled: bool,
    scale: f32,
    finite_steps: usize,
}

impl Fp16LossScaler {
    fn new(mixed_precision: MixedPrecision) -> Self {
        Self {
            enabled: mixed_precision == MixedPrecision::Fp16,
            scale: FP16_INITIAL_LOSS_SCALE,
            finite_steps: 0,
        }
    }

    fn scale(&self) -> f32 {
        if self.enabled { self.scale } else { 1.0 }
    }

    fn scale_loss<B: AutodiffBackend>(&self, loss: Tensor<B, 1>) -> Tensor<B, 1> {
        if self.enabled {
            loss * self.scale
        } else {
            loss
        }
    }

    fn unscale_and_check<B, M>(&self, grads: GradientsParams, model: &M) -> (GradientsParams, bool)
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
    {
        if !self.enabled {
            return (grads, true);
        }
        let mut visitor = GradientsFiniteVisitor::<B> {
            input: grads,
            output: GradientsParams::new(),
            inv_scale: 1.0 / self.scale,
            finite: true,
            backend: PhantomData,
        };
        model.visit(&mut visitor);
        (visitor.output, visitor.finite)
    }

    fn backoff(&mut self) {
        if !self.enabled {
            return;
        }
        self.scale = (self.scale * 0.5).max(FP16_MIN_LOSS_SCALE);
        self.finite_steps = 0;
    }

    fn update_after_success(&mut self) {
        if !self.enabled {
            return;
        }
        self.finite_steps += 1;
        if self.finite_steps >= FP16_LOSS_SCALE_GROWTH_INTERVAL {
            self.scale = (self.scale * 2.0).min(FP16_MAX_LOSS_SCALE);
            self.finite_steps = 0;
        }
    }
}

struct GradientsFiniteVisitor<B: AutodiffBackend> {
    input: GradientsParams,
    output: GradientsParams,
    inv_scale: f32,
    finite: bool,
    backend: PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradientsFiniteVisitor<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let Some(grad) = self.input.remove::<B::InnerBackend, D>(param.id) else {
            return;
        };
        let grad = grad * self.inv_scale;
        if !bool_scalar_tensor_value(grad.clone().is_finite().all()) {
            self.finite = false;
        }
        self.output.register::<B::InnerBackend, D>(param.id, grad);
    }
}

fn bool_scalar_tensor_value<B: burn::tensor::backend::Backend>(tensor: Tensor<B, 1, Bool>) -> bool {
    let data = tensor.into_data();
    match data.dtype {
        DType::Bool(BoolStore::Native) => data
            .to_vec::<bool>()
            .expect("scalar tensor should be native bool")
            .first()
            .copied()
            .expect("scalar tensor should contain one value"),
        DType::Bool(BoolStore::U8) => {
            data.to_vec::<u8>()
                .expect("scalar tensor should be u8 bool")
                .first()
                .copied()
                .expect("scalar tensor should contain one value")
                != 0
        }
        DType::Bool(BoolStore::U32) => {
            data.to_vec::<u32>()
                .expect("scalar tensor should be u32 bool")
                .first()
                .copied()
                .expect("scalar tensor should contain one value")
                != 0
        }
        dtype => panic!("scalar tensor should be bool, got {dtype:?}"),
    }
}

impl TrainProfilingStep {
    fn new(
        epoch: usize,
        step: usize,
        epoch_batch: usize,
        data_time: f32,
        wait_time: Duration,
        gpu_timing_mode: &'static str,
        profiling_ablation: ProfilingAblation,
    ) -> Self {
        Self {
            record_type: "training_profile_step",
            epoch,
            step,
            epoch_batch,
            dataloader_data_time: data_time,
            dataloader_wait_time: wait_time.as_secs_f32(),
            data_time,
            wait_time: wait_time.as_secs_f32(),
            gpu_timing_mode,
            profiling_ablation,
            ..Self::default()
        }
    }
}

impl TrainProfiler {
    fn push(&mut self, step: TrainProfilingStep) {
        self.steps.push(step);
    }

    fn warm_average(&self) -> TrainProfilingAverage {
        let warm_steps = self.steps.iter().skip(1).collect::<Vec<_>>();
        let count = warm_steps.len().max(1) as f32;
        let sum = |field: fn(&TrainProfilingStep) -> f32| -> f32 {
            warm_steps.iter().map(|step| field(step)).sum::<f32>() / count
        };
        TrainProfilingAverage {
            record_type: "training_profile_warm_average",
            skipped_cold_steps: self.steps.len().min(1),
            warm_steps: warm_steps.len(),
            batch_wall_time: sum(|step| step.batch_wall_time),
            dataloader_data_time: sum(|step| step.dataloader_data_time),
            dataloader_wait_time: sum(|step| step.dataloader_wait_time),
            h2d_copy_cpu_time: sum(|step| step.h2d_copy_cpu_time),
            h2d_image_tensor_cpu_time: sum(|step| step.h2d_image_tensor_cpu_time),
            forward_cpu_time: sum(|step| step.forward_cpu_time),
            target_tensor_cpu_time: sum(|step| step.target_tensor_cpu_time),
            loss_compute_cpu_time: sum(|step| step.loss_compute_cpu_time),
            backward_cpu_time: sum(|step| step.backward_cpu_time),
            optimizer_step_cpu_time: sum(|step| step.optimizer_step_cpu_time),
            h2d_copy_gpu_time: sum(|step| step.h2d_copy_gpu_time),
            h2d_image_tensor_gpu_time: sum(|step| step.h2d_image_tensor_gpu_time),
            h2d_target_tensor_gpu_time: sum(|step| step.h2d_target_tensor_gpu_time),
            forward_gpu_time: sum(|step| step.forward_gpu_time),
            loss_compute_gpu_time: sum(|step| step.loss_compute_gpu_time),
            backward_gpu_time: sum(|step| step.backward_gpu_time),
            optimizer_step_gpu_time: sum(|step| step.optimizer_step_gpu_time),
            target_gt_text_requires_grad: warm_steps
                .iter()
                .any(|step| step.target_gt_text_requires_grad),
            target_gt_kernel_requires_grad: warm_steps
                .iter()
                .any(|step| step.target_gt_kernel_requires_grad),
            target_training_mask_requires_grad: warm_steps
                .iter()
                .any(|step| step.target_training_mask_requires_grad),
            max_backward_call_count: warm_steps
                .iter()
                .map(|step| step.backward_call_count)
                .max()
                .unwrap_or(0),
            profiling_ablation: self
                .steps
                .first()
                .map(|step| step.profiling_ablation)
                .unwrap_or(ProfilingAblation::Normal),
            loss_to_cpu_breakdown_time: sum(|step| step.loss_to_cpu_breakdown_time),
            metrics_write_time: sum(|step| step.metrics_write_time),
            backend_sync_time: sum(|step| step.backend_sync_time),
            gpu_timing_mode: self
                .steps
                .first()
                .map(|step| step.gpu_timing_mode)
                .unwrap_or("wall_time"),
            batch_time: sum(|step| step.batch_time),
            data_time: sum(|step| step.data_time),
            wait_time: sum(|step| step.wait_time),
        }
    }
}

impl ProfilingAblation {
    fn needs_loss(self) -> bool {
        !matches!(
            self,
            ProfilingAblation::ForwardOnly | ProfilingAblation::DummyScalarBackward
        )
    }

    fn needs_backward(self) -> bool {
        !matches!(
            self,
            ProfilingAblation::ForwardOnly | ProfilingAblation::ForwardLossOnly
        )
    }

    fn needs_optimizer(self) -> bool {
        matches!(self, ProfilingAblation::Normal)
    }
}

fn loss_component_selection(ablation: ProfilingAblation) -> LossComponentSelection {
    match ablation {
        ProfilingAblation::RegionBceOnly => LossComponentSelection {
            region_bce: true,
            kernel_bce: false,
            region_dice: false,
            kernel_dice: false,
        },
        ProfilingAblation::KernelBceOnly => LossComponentSelection {
            region_bce: false,
            kernel_bce: true,
            region_dice: false,
            kernel_dice: false,
        },
        ProfilingAblation::RegionDiceOnly => LossComponentSelection {
            region_bce: false,
            kernel_bce: false,
            region_dice: true,
            kernel_dice: false,
        },
        ProfilingAblation::KernelDiceOnly => LossComponentSelection {
            region_bce: false,
            kernel_bce: false,
            region_dice: false,
            kernel_dice: true,
        },
        ProfilingAblation::BceOnly => LossComponentSelection {
            region_bce: true,
            kernel_bce: true,
            region_dice: false,
            kernel_dice: false,
        },
        ProfilingAblation::DiceOnly => LossComponentSelection {
            region_bce: false,
            kernel_bce: false,
            region_dice: true,
            kernel_dice: true,
        },
        ProfilingAblation::RegionOnly => LossComponentSelection {
            region_bce: true,
            kernel_bce: false,
            region_dice: true,
            kernel_dice: false,
        },
        ProfilingAblation::KernelOnly => LossComponentSelection {
            region_bce: false,
            kernel_bce: true,
            region_dice: false,
            kernel_dice: true,
        },
        _ => LossComponentSelection::ALL,
    }
}

fn profile_backward_loss<B: AutodiffBackend>(
    ablation: ProfilingAblation,
    output: &crate::model::ModelOutput<B>,
    loss: Option<&crate::loss::TensorLossBreakdown<B>>,
) -> Option<Tensor<B, 1>> {
    match ablation {
        ProfilingAblation::DummyScalarBackward => {
            let pixels = output.text_region_logits.dims().iter().product::<usize>() as f32;
            Some(
                (output.text_region_logits.clone().sum() + output.kernel_logits.clone().sum())
                    / pixels.max(1.0)
                    * 1e-3,
            )
        }
        _ => loss.map(|loss| loss.total_loss.clone()),
    }
}

fn gpu_timing_mode(config: &TrainConfig, profile_timing: impl TrainProfileTiming) -> &'static str {
    match config.backend {
        crate::config::BackendKind::Cuda if config.profiling_enabled => profile_timing.mode(),
        _ => "wall_time",
    }
}

fn print_profile_average(average: &TrainProfilingAverage) {
    println!(
        "profiling_warm_average skipped_cold_steps={} warm_steps={} batch_wall_time={:.5} dataloader_data_time={:.5} dataloader_wait_time={:.5} h2d_copy_cpu_time={:.5} forward_cpu_time={:.5} target_tensor_cpu_time={:.5} loss_compute_cpu_time={:.5} backward_cpu_time={:.5} optimizer_step_cpu_time={:.5} h2d_copy_gpu_time={:.5} forward_gpu_time={:.5} loss_compute_gpu_time={:.5} backward_gpu_time={:.5} optimizer_step_gpu_time={:.5} loss_to_cpu_breakdown_time={:.5} metrics_write_time={:.5} gpu_timing_mode={}",
        average.skipped_cold_steps,
        average.warm_steps,
        average.batch_wall_time,
        average.dataloader_data_time,
        average.dataloader_wait_time,
        average.h2d_copy_cpu_time,
        average.forward_cpu_time,
        average.target_tensor_cpu_time,
        average.loss_compute_cpu_time,
        average.backward_cpu_time,
        average.optimizer_step_cpu_time,
        average.h2d_copy_gpu_time,
        average.forward_gpu_time,
        average.loss_compute_gpu_time,
        average.backward_gpu_time,
        average.optimizer_step_gpu_time,
        average.loss_to_cpu_breakdown_time,
        average.metrics_write_time,
        average.gpu_timing_mode,
    );
}

struct PrefetchedBatch {
    batch: TrainingBatch,
    data_time: f32,
    epoch_batch: usize,
    samples_processed: usize,
    samples_total: usize,
}

type PrefetchMessage = Result<PrefetchedBatch>;
type PrefetchWorker = JoinHandle<Result<()>>;

fn spawn_epoch_prefetch(
    dataset: Arc<SubtitleDataset>,
    config: TrainConfig,
) -> (Receiver<PrefetchMessage>, PrefetchWorker) {
    let capacity = config.prefetch_batches.clamp(1, 2);
    let (sender, receiver) = sync_channel(capacity);
    let worker = thread::spawn(move || -> Result<()> {
        for chunk_start in (0..dataset.len()).step_by(config.batch_size.max(1)) {
            let chunk_end = (chunk_start + config.batch_size).min(dataset.len());
            let epoch_batch = chunk_start / config.batch_size.max(1) + 1;
            let data_start = Instant::now();
            let batch = prepare_training_batch(&dataset, &config, chunk_start);
            let message = batch.map(|batch| {
                batch.map(|batch| PrefetchedBatch {
                    batch,
                    data_time: data_start.elapsed().as_secs_f32(),
                    epoch_batch,
                    samples_processed: chunk_end,
                    samples_total: dataset.len(),
                })
            });
            match message {
                Ok(Some(prepared)) => sender
                    .send(Ok(prepared))
                    .map_err(|_| anyhow!("training prefetch receiver dropped"))?,
                Ok(None) => {}
                Err(err) => {
                    let text = format!("{err:#}");
                    let _ = sender.send(Err(err));
                    return Err(anyhow!("training prefetch failed: {text}"));
                }
            }
        }
        Ok(())
    });
    (receiver, worker)
}

fn prepare_training_batch(
    dataset: &SubtitleDataset,
    config: &TrainConfig,
    chunk_start: usize,
) -> Result<Option<TrainingBatch>> {
    let chunk_end = (chunk_start + config.batch_size).min(dataset.len());
    let mut samples = Vec::with_capacity(chunk_end - chunk_start);
    for index in chunk_start..chunk_end {
        let sample = dataset.load_sample(index)?;
        if sample.ignored {
            continue;
        }
        samples.push(preprocess_sample(&sample, config, true)?);
    }
    if samples.is_empty() {
        return Ok(None);
    }
    Ok(Some(collate_batch_with_config(samples, config)))
}

fn next_prefetched_batch(
    receiver: &Receiver<PrefetchMessage>,
    worker: &PrefetchWorker,
) -> Result<Option<PrefetchedBatch>> {
    match receiver.recv() {
        Ok(Ok(batch)) => Ok(Some(batch)),
        Ok(Err(err)) => Err(err),
        Err(_) if worker.is_finished() => Ok(None),
        Err(err) => Err(anyhow!(
            "training prefetch channel closed unexpectedly: {err}"
        )),
    }
}

fn join_prefetch_worker(worker: PrefetchWorker) -> Result<()> {
    worker
        .join()
        .map_err(|_| anyhow!("training prefetch worker panicked"))?
        .context("training prefetch worker failed")
}

fn save_training_checkpoint<B, O>(
    path: impl AsRef<Path>,
    model: SubFastNet<B::InnerBackend>,
    optimizer: &O,
    meta: &CheckpointMeta,
) -> Result<()>
where
    B: burn::tensor::backend::AutodiffBackend,
    O: burn::optim::Optimizer<SubFastNet<B>, B>,
{
    let path = path.as_ref();
    save_checkpoint(path, model, meta)?;
    save_optimizer_record::<O, B>(path, optimizer)?;
    Ok(())
}

fn print_training_status(config: &TrainConfig, metrics: &TrainStepMetrics, tui_active: bool) {
    if !config.tui_enabled || !tui_active {
        println!(
            "epoch={}/{} batch={}/{} samples={}/{} step={} total_loss={:.5} region={:.5} kernel={:.5} samples/s={:.2}",
            metrics.epoch,
            config.epochs,
            metrics.epoch_batch,
            metrics.epoch_batches,
            metrics.epoch_samples_processed,
            metrics.epoch_samples_total,
            metrics.step,
            metrics.total_loss,
            metrics.region_loss,
            metrics.kernel_loss,
            metrics.samples_per_second
        );
    }
}

fn print_validation_status(
    config: &TrainConfig,
    epoch: usize,
    step: usize,
    validation: &crate::validate::ValidationSummary,
    tui_active: bool,
) {
    if !config.tui_enabled || !tui_active {
        println!(
            "epoch={epoch} step={step} val_loss={:.5} precision={:.4} recall={:.4} f1={:.4} mean_iou={:.4} fps={:.2} latency_p50={:.2} latency_p95={:.2} postprocess_latency={:.2}",
            validation.val_loss,
            validation.precision,
            validation.recall,
            validation.f1,
            validation.mean_iou,
            validation.fps,
            validation.latency_p50,
            validation.latency_p95,
            validation.postprocess_latency
        );
    }
}

fn scheduled_learning_rate(config: &TrainConfig, epoch: usize) -> f32 {
    config.learning_rate * config.scheduler_gamma.powi(epoch.saturating_sub(1) as i32)
}
