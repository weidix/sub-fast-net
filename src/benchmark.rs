use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

use anyhow::Result;
use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::backend::{AutodiffBackend, Backend},
    train::metric::{CpuMemory, Metric, MetricMetadata},
};
use serde::Serialize;

use crate::{
    checkpoint::load_checkpoint_model,
    config::TrainConfig,
    dataset::SubtitleDataset,
    loss::{LossTensorCache, compute_loss, compute_tensor_loss_breakdown_cached},
    metrics::percentile,
    postprocess::PostprocessConfig,
    preprocess::{collate_batch_with_config, preprocess_sample},
    validate::{downsample_masks, feature_metas},
};

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkSummary {
    pub record_type: &'static str,
    pub dataloader_throughput: f32,
    pub preprocess_throughput: f32,
    pub train_step_time: Option<f32>,
    pub validation_step_time: f32,
    pub inference_fps: f32,
    pub latency_p50: f32,
    pub latency_p95: f32,
    pub postprocess_latency: f32,
    pub end_to_end_latency: f32,
    pub preprocess_time: f32,
    pub preprocess_latency_p50: f32,
    pub preprocess_latency_p95: f32,
    pub forward_time: f32,
    pub forward_latency_p50: f32,
    pub forward_latency_p95: f32,
    pub postprocess_time: f32,
    pub postprocess_latency_p50: f32,
    pub postprocess_latency_p95: f32,
    pub candidate_count: usize,
    pub final_box_count: usize,
    pub max_region_probability: f32,
    pub max_kernel_probability: f32,
    pub train_step_time_note: String,
    pub memory_usage: String,
}

pub fn benchmark(config: &TrainConfig, checkpoint: &str) -> Result<BenchmarkSummary> {
    crate::backend::benchmark_with_configured_backend(config, checkpoint)
}

pub fn benchmark_train_step_with_configured_backend(config: &TrainConfig) -> Result<f32> {
    match config.backend {
        crate::config::BackendKind::Cpu => {
            benchmark_train_step_backend::<crate::backend::CpuAutodiffBackend>(config)
        }
        crate::config::BackendKind::Cuda => benchmark_train_step_cuda(config),
        crate::config::BackendKind::Wgpu => benchmark_train_step_wgpu(config),
    }
}

pub fn benchmark_backend<B: Backend>(
    config: &TrainConfig,
    checkpoint: &str,
) -> Result<BenchmarkSummary> {
    let device = B::Device::default();
    let (model, _checkpoint_meta) = load_checkpoint_model::<B>(checkpoint, &device)?;
    let dataset = SubtitleDataset::from_val_config(config)?;
    let sample_count = dataset
        .len()
        .min(config.max_val_samples.unwrap_or(32).max(1));
    let post_config = PostprocessConfig {
        threshold_region: config.threshold_region,
        threshold_kernel: config.threshold_kernel,
        min_width: config.min_kernel_width as f32,
        min_height: config.min_kernel_height as f32,
    };
    let load_start = Instant::now();
    let mut samples = Vec::new();
    for index in 0..sample_count {
        let sample = dataset.load_sample(index)?;
        if !sample.ignored {
            samples.push(sample);
        }
    }
    let load_time = load_start.elapsed().as_secs_f32();
    let mut preprocess_latencies = Vec::new();
    let mut forward_latencies = Vec::new();
    let mut postprocess_latencies = Vec::new();
    let mut validation_latencies = Vec::new();
    let mut end_to_end_latencies = Vec::new();
    let mut candidate_count = 0;
    let mut final_box_count = 0;
    let mut max_region_probability = 0.0_f32;
    let mut max_kernel_probability = 0.0_f32;
    for sample in &samples {
        let preprocess_start = Instant::now();
        let preprocessed = preprocess_sample(sample, config, false)?;
        let preprocess_time = preprocess_start.elapsed().as_secs_f32();
        let batch = collate_batch_with_config(vec![preprocessed], config);
        let forward_start = Instant::now();
        let output = crate::model::output_to_cpu(model.forward(batch.image_tensor(&device)));
        let forward_time = forward_start.elapsed().as_secs_f32();
        max_region_probability =
            max_region_probability.max(max_sigmoid(&output.text_region_logits));
        max_kernel_probability = max_kernel_probability.max(max_sigmoid(&output.kernel_logits));
        let post_start = Instant::now();
        let metas = feature_metas(
            &batch.img_metas,
            batch.width,
            batch.height,
            output.width,
            output.height,
        );
        let post_results =
            crate::postprocess::postprocess_output_with_stats(&output, &metas, &post_config);
        let post_time = post_start.elapsed().as_secs_f32();
        for result in &post_results {
            candidate_count += result.candidate_count;
            final_box_count += result.final_box_count;
        }
        let validation_start = Instant::now();
        let _loss = compute_loss(
            &output,
            &downsample_masks(
                &batch.gt_texts,
                batch.height,
                batch.width,
                output.height,
                output.width,
            ),
            &downsample_masks(
                &batch.gt_kernels,
                batch.height,
                batch.width,
                output.height,
                output.width,
            ),
            &downsample_masks(
                &batch.training_masks,
                batch.height,
                batch.width,
                output.height,
                output.width,
            ),
        );
        let validation_time = validation_start.elapsed().as_secs_f32() + forward_time + post_time;
        preprocess_latencies.push(preprocess_time * 1000.0);
        forward_latencies.push(forward_time * 1000.0);
        postprocess_latencies.push(post_time * 1000.0);
        validation_latencies.push(validation_time * 1000.0);
        end_to_end_latencies.push((preprocess_time + forward_time + post_time) * 1000.0);
    }
    let evaluated_count = end_to_end_latencies.len();
    let total_end_to_end_ms = end_to_end_latencies.iter().sum::<f32>().max(1e-6);
    let total_preprocess_ms = preprocess_latencies.iter().sum::<f32>();
    let total_forward_ms = forward_latencies.iter().sum::<f32>();
    let total_postprocess_ms = postprocess_latencies.iter().sum::<f32>();
    let mut tmp = end_to_end_latencies.clone();
    let latency_p50 = percentile(&mut tmp, 0.50);
    let mut tmp = end_to_end_latencies.clone();
    let latency_p95 = percentile(&mut tmp, 0.95);
    let mut tmp = preprocess_latencies.clone();
    let preprocess_latency_p50 = percentile(&mut tmp, 0.50);
    let mut tmp = preprocess_latencies.clone();
    let preprocess_latency_p95 = percentile(&mut tmp, 0.95);
    let mut tmp = forward_latencies.clone();
    let forward_latency_p50 = percentile(&mut tmp, 0.50);
    let mut tmp = forward_latencies.clone();
    let forward_latency_p95 = percentile(&mut tmp, 0.95);
    let mut tmp = postprocess_latencies.clone();
    let postprocess_latency_p50 = percentile(&mut tmp, 0.50);
    let mut tmp = postprocess_latencies.clone();
    let postprocess_latency_p95 = percentile(&mut tmp, 0.95);
    let mut tmp = validation_latencies.clone();
    let validation_step_time = percentile(&mut tmp, 0.50);
    let train_step_time = benchmark_train_step_with_configured_backend(config)?;
    let memory_usage = burn_memory_usage_note();
    let summary = BenchmarkSummary {
        record_type: "benchmark",
        dataloader_throughput: samples.len() as f32 / load_time.max(1e-6),
        preprocess_throughput: evaluated_count as f32 / (total_preprocess_ms / 1000.0).max(1e-6),
        train_step_time: Some(train_step_time),
        validation_step_time,
        inference_fps: evaluated_count as f32 / (total_end_to_end_ms / 1000.0),
        latency_p50,
        latency_p95,
        postprocess_latency: total_postprocess_ms / evaluated_count.max(1) as f32,
        end_to_end_latency: total_end_to_end_ms / evaluated_count.max(1) as f32,
        preprocess_time: total_preprocess_ms / evaluated_count.max(1) as f32,
        preprocess_latency_p50,
        preprocess_latency_p95,
        forward_time: total_forward_ms / evaluated_count.max(1) as f32,
        forward_latency_p50,
        forward_latency_p95,
        postprocess_time: total_postprocess_ms / evaluated_count.max(1) as f32,
        postprocess_latency_p50,
        postprocess_latency_p95,
        candidate_count,
        final_box_count,
        max_region_probability,
        max_kernel_probability,
        train_step_time_note:
            "measured with one real autodiff forward/loss/backward/Adam optimizer step".to_string(),
        memory_usage,
    };
    fs::create_dir_all(&config.output_dir)?;
    fs::write(
        Path::new(&config.output_dir).join("benchmark_summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    fs::write(
        Path::new(&config.output_dir).join("benchmark_metrics.jsonl"),
        format!("{}\n", serde_json::to_string(&summary)?),
    )?;
    append_standard_benchmark_outputs(config, &summary)?;
    Ok(summary)
}

fn max_sigmoid(logits: &[Vec<f32>]) -> f32 {
    logits
        .iter()
        .flat_map(|values| values.iter().copied())
        .map(crate::model::sigmoid)
        .fold(0.0_f32, f32::max)
}

fn append_standard_benchmark_outputs(
    config: &TrainConfig,
    summary: &BenchmarkSummary,
) -> Result<()> {
    let output_dir = Path::new(&config.output_dir);
    let mut metrics = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_dir.join("metrics.jsonl"))?;
    writeln!(metrics, "{}", serde_json::to_string(summary)?)?;

    let summary_path = output_dir.join("summary.json");
    let mut root = if summary_path.is_file() {
        serde_json::from_str::<serde_json::Value>(&fs::read_to_string(&summary_path)?)?
    } else {
        serde_json::json!({})
    };
    match &mut root {
        serde_json::Value::Object(map) => {
            map.insert("benchmark".to_string(), serde_json::to_value(summary)?);
        }
        other => {
            *other = serde_json::json!({
                "previous_summary": other.clone(),
                "benchmark": summary,
            });
        }
    }
    fs::write(summary_path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

pub fn benchmark_train_step_backend<B: AutodiffBackend>(config: &TrainConfig) -> Result<f32> {
    let device = B::Device::default();
    B::seed(&device, config.seed);
    let dataset = SubtitleDataset::from_train_config(config)?;
    let mut samples = Vec::new();
    for index in 0..dataset.len() {
        let sample = dataset.load_sample(index)?;
        if sample.ignored {
            continue;
        }
        samples.push(preprocess_sample(&sample, config, true)?);
        if samples.len() >= config.batch_size.max(1) {
            break;
        }
    }
    if samples.is_empty() {
        anyhow::bail!("cannot benchmark train_step_time: no usable training samples");
    }
    let batch = collate_batch_with_config(samples, config);
    let mut model = crate::model::SubFastNet::<B>::new(config.model_variant, &device);
    let mut optimizer = AdamConfig::new().init::<B, crate::model::SubFastNet<B>>();
    let mut loss_tensor_cache = LossTensorCache::<B>::new(&device);
    let start = Instant::now();
    let images = batch.image_tensor::<B>(&device);
    let output = model.forward(images);
    let gt_text = batch.text_tensor::<B>(output.height, output.width, &device);
    let gt_kernel = batch.kernel_tensor::<B>(output.height, output.width, &device);
    let training_mask = batch.training_mask_tensor::<B>(output.height, output.width, &device);
    let loss = compute_tensor_loss_breakdown_cached(
        &output,
        gt_text,
        gt_kernel,
        training_mask,
        &mut loss_tensor_cache,
    );
    let grads = GradientsParams::from_grads(loss.total_loss.backward(), &model);
    model = optimizer.step(config.learning_rate as f64, model, grads);
    let _ = model;
    Ok(start.elapsed().as_secs_f32() * 1000.0)
}

fn burn_memory_usage_note() -> String {
    let mut metric = CpuMemory::new();
    let metadata = MetricMetadata {
        progress: burn::data::dataloader::Progress {
            items_processed: 1,
            items_total: 1,
        },
        global_progress: burn::data::dataloader::Progress {
            items_processed: 1,
            items_total: 1,
        },
        iteration: Some(1),
        lr: None,
    };
    let cpu = metric.update(&(), &metadata).formatted;
    format!(
        "{cpu}; backend/GPU allocator memory unsupported: Burn 0.21 exposes sys-metrics CPU memory, but no stable public backend memory usage API on Backend"
    )
}

#[cfg(feature = "backend-cuda")]
fn benchmark_train_step_cuda(config: &TrainConfig) -> Result<f32> {
    benchmark_train_step_backend::<crate::backend::CudaAutodiffBackend>(config)
}

#[cfg(not(feature = "backend-cuda"))]
fn benchmark_train_step_cuda(_config: &TrainConfig) -> Result<f32> {
    anyhow::bail!("backend=cuda requires building with feature backend-cuda")
}

#[cfg(feature = "backend-wgpu")]
fn benchmark_train_step_wgpu(config: &TrainConfig) -> Result<f32> {
    benchmark_train_step_backend::<crate::backend::WgpuAutodiffBackend>(config)
}

#[cfg(not(feature = "backend-wgpu"))]
fn benchmark_train_step_wgpu(_config: &TrainConfig) -> Result<f32> {
    anyhow::bail!("backend=wgpu requires building with feature backend-wgpu")
}
