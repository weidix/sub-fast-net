use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

use anyhow::Result;
use burn::{module::Module, tensor::backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    config::TrainConfig,
    dataset::SubtitleDataset,
    loss::{LossBreakdown, compute_loss},
    metrics::{DetectionMetricsAccumulator, match_detection_metrics, percentile},
    model::{SubFastNet, output_to_cpu},
    postprocess::{PostprocessConfig, postprocess_output},
    preprocess::{ImageMeta, PixelBox, collate_batch_with_config, preprocess_sample},
};

#[derive(Debug, Clone, Default, Serialize)]
pub struct ValidationSummary {
    pub val_loss: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub mean_iou: f32,
    pub fps: f32,
    pub latency_p50: f32,
    pub latency_p95: f32,
    pub postprocess_latency: f32,
    pub false_positive_count: usize,
    pub false_negative_count: usize,
    pub ignored_sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub image: String,
    pub source: Option<String>,
    pub frame_id: Option<String>,
    pub width: u32,
    pub height: u32,
    pub pred_boxes: Vec<PixelBox>,
    pub gt_boxes: Vec<PixelBox>,
    pub iou: Vec<f32>,
    pub reason: String,
}

pub fn validate_model(
    config: &TrainConfig,
    dataset: &SubtitleDataset,
    model: &SubFastNet<impl Backend>,
) -> Result<ValidationSummary> {
    let device = model
        .devices()
        .into_iter()
        .next()
        .expect("model should have a device");
    let mut losses = Vec::new();
    let mut aggregate = DetectionMetricsAccumulator::default();
    let mut latencies = Vec::new();
    let mut postprocess_latencies = Vec::new();
    let mut ignored_sample_count = 0;
    let mut evaluated_sample_count = 0;
    let post_config = PostprocessConfig {
        threshold_region: config.threshold_region,
        threshold_kernel: config.threshold_kernel,
        min_width: config.min_kernel_width as f32,
        min_height: config.min_kernel_height as f32,
        max_width_ratio: config.max_detection_width_ratio,
    };
    for chunk_start in (0..dataset.len()).step_by(config.batch_size.max(1)) {
        let chunk_end = (chunk_start + config.batch_size).min(dataset.len());
        let mut samples = Vec::new();
        for index in chunk_start..chunk_end {
            let sample = match dataset.load_sample(index) {
                Ok(sample) => sample,
                Err(_) => {
                    ignored_sample_count += 1;
                    continue;
                }
            };
            if sample.ignored {
                ignored_sample_count += 1;
                continue;
            }
            match preprocess_sample(&sample, config, false) {
                Ok(sample) => samples.push(sample),
                Err(_) => ignored_sample_count += 1,
            }
        }
        if samples.is_empty() {
            continue;
        }
        let batch = collate_batch_with_config(samples, config);
        let forward_start = Instant::now();
        let output = output_to_cpu(model.forward(batch.image_tensor(&device)));
        let forward_ms = forward_start.elapsed().as_secs_f32() * 1000.0;
        let metas = feature_metas(
            &batch.img_metas,
            batch.width,
            batch.height,
            output.width,
            output.height,
        );
        let post_start = Instant::now();
        let postprocess_results =
            crate::postprocess::postprocess_output_with_stats(&output, &metas, &post_config);
        let measured_post_ms = post_start.elapsed().as_secs_f32() * 1000.0;
        let reported_post_ms = postprocess_results
            .iter()
            .map(|result| result.postprocess_latency_ms)
            .sum::<f32>();
        let post_ms = measured_post_ms.max(reported_post_ms);
        let detections = postprocess_results
            .iter()
            .map(|result| result.boxes.clone())
            .collect::<Vec<_>>();
        postprocess_latencies.push(post_ms);
        latencies.push(forward_ms + post_ms);
        let loss = compute_loss(
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
        losses.push(loss.total_loss);
        for ((predicted, gt), meta) in detections.iter().zip(&batch.gt_boxes).zip(&batch.img_metas)
        {
            let pred_boxes = predicted
                .iter()
                .map(|bbox| crate::preprocess::PixelBox {
                    x1: bbox.x1,
                    y1: bbox.y1,
                    x2: bbox.x2,
                    y2: bbox.y2,
                })
                .collect::<Vec<_>>();
            let gt_boxes = restore_gt_boxes_to_output_space(gt, meta);
            let metrics = match_detection_metrics(&pred_boxes, &gt_boxes, config.iou_threshold);
            aggregate.add(&metrics);
            evaluated_sample_count += 1;
        }
    }
    let mut latency_copy = latencies.clone();
    let latency_p50 = percentile(&mut latency_copy, 0.50);
    let mut latency_copy = latencies.clone();
    let latency_p95 = percentile(&mut latency_copy, 0.95);
    let total_ms = latencies.iter().sum::<f32>().max(1e-6);
    Ok(ValidationSummary {
        val_loss: average(&losses),
        precision: aggregate.precision(),
        recall: aggregate.recall(),
        f1: aggregate.f1(),
        mean_iou: aggregate.mean_iou(),
        fps: evaluated_sample_count as f32 / (total_ms / 1000.0),
        latency_p50,
        latency_p95,
        postprocess_latency: average(&postprocess_latencies),
        false_positive_count: aggregate.false_positive_count,
        false_negative_count: aggregate.false_negative_count,
        ignored_sample_count,
    })
}

pub fn feature_metas(
    metas: &[crate::preprocess::ImageMeta],
    input_w: usize,
    input_h: usize,
    output_w: usize,
    output_h: usize,
) -> Vec<crate::preprocess::ImageMeta> {
    let sx = output_w as f32 / input_w.max(1) as f32;
    let sy = output_h as f32 / input_h.max(1) as f32;
    metas
        .iter()
        .cloned()
        .map(|mut meta| {
            meta.scale *= sx.min(sy);
            meta.resized_width = (meta.resized_width as f32 * sx).round() as u32;
            meta.resized_height = (meta.resized_height as f32 * sy).round() as u32;
            meta
        })
        .collect()
}

pub fn downsample_masks(
    masks: &[Vec<f32>],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<Vec<f32>> {
    masks
        .iter()
        .map(|mask| {
            let mut output = Vec::with_capacity(dst_h * dst_w);
            for y in 0..dst_h {
                let src_y = (y * src_h / dst_h).min(src_h - 1);
                for x in 0..dst_w {
                    let src_x = (x * src_w / dst_w).min(src_w - 1);
                    output.push(mask[src_y * src_w + src_x]);
                }
            }
            output
        })
        .collect()
}

pub fn write_error_reports(
    config: &TrainConfig,
    dataset: &SubtitleDataset,
    model: &SubFastNet<impl Backend>,
) -> Result<()> {
    let dir = Path::new(&config.output_dir).join("errors");
    fs::create_dir_all(&dir)?;
    let mut false_positive = Vec::new();
    let mut false_negative = Vec::new();
    let mut low_iou = Vec::new();
    let mut ignored = Vec::new();
    let device = model
        .devices()
        .into_iter()
        .next()
        .expect("model should have a device");
    let post_config = PostprocessConfig {
        threshold_region: config.threshold_region,
        threshold_kernel: config.threshold_kernel,
        min_width: config.min_kernel_width as f32,
        min_height: config.min_kernel_height as f32,
        max_width_ratio: config.max_detection_width_ratio,
    };
    for index in 0..dataset.len() {
        let sample = match dataset.load_sample(index) {
            Ok(sample) => sample,
            Err(err) => {
                ignored.push(ErrorRecord {
                    image: String::new(),
                    source: None,
                    frame_id: None,
                    width: 0,
                    height: 0,
                    pred_boxes: Vec::new(),
                    gt_boxes: Vec::new(),
                    iou: Vec::new(),
                    reason: format!("ignored: {err}"),
                });
                continue;
            }
        };
        if sample.ignored {
            ignored.push(error_record_from_sample(
                &sample,
                Vec::new(),
                sample.pixel_boxes_after_label_masks.clone(),
                Vec::new(),
                sample.ignore_reason.as_deref().unwrap_or("ignored"),
            ));
            continue;
        }
        if !sample.issues.is_empty() {
            ignored.push(error_record_from_sample(
                &sample,
                Vec::new(),
                sample.pixel_boxes_after_label_masks.clone(),
                Vec::new(),
                "warning",
            ));
        }
        let preprocessed = match preprocess_sample(&sample, config, false) {
            Ok(sample) => sample,
            Err(err) => {
                ignored.push(error_record_from_sample(
                    &sample,
                    Vec::new(),
                    sample.pixel_boxes_after_label_masks.clone(),
                    Vec::new(),
                    &format!("ignored: preprocess failed: {err}"),
                ));
                continue;
            }
        };
        let batch = collate_batch_with_config(vec![preprocessed], config);
        let output = crate::model::output_to_cpu(model.forward(batch.image_tensor(&device)));
        let metas = feature_metas(
            &batch.img_metas,
            batch.width,
            batch.height,
            output.width,
            output.height,
        );
        let pred = postprocess_output(&output, &metas, &post_config)
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .map(|bbox| PixelBox {
                x1: bbox.x1,
                y1: bbox.y1,
                x2: bbox.x2,
                y2: bbox.y2,
            })
            .collect::<Vec<_>>();
        let gt = batch
            .gt_boxes
            .first()
            .map(|boxes| restore_gt_boxes_to_output_space(boxes, &batch.img_metas[0]))
            .unwrap_or_default();
        let ious = best_ious(&pred, &gt);
        let metrics = match_detection_metrics(&pred, &gt, config.iou_threshold);
        if metrics.false_positive_count > 0 {
            false_positive.push(error_record_from_sample(
                &sample,
                pred.clone(),
                gt.clone(),
                ious.clone(),
                "false_positive",
            ));
        }
        if metrics.false_negative_count > 0 {
            false_negative.push(error_record_from_sample(
                &sample,
                pred.clone(),
                gt.clone(),
                ious.clone(),
                "false_negative",
            ));
        }
        if !pred.is_empty()
            && !gt.is_empty()
            && ious.iter().copied().fold(0.0_f32, f32::max) < config.iou_threshold
        {
            low_iou.push(error_record_from_sample(&sample, pred, gt, ious, "low_iou"));
        }
    }
    write_jsonl(&dir.join("false_positive.jsonl"), &false_positive)?;
    write_jsonl(&dir.join("false_negative.jsonl"), &false_negative)?;
    write_jsonl(&dir.join("low_iou.jsonl"), &low_iou)?;
    write_jsonl(&dir.join("ignored_samples.jsonl"), &ignored)?;
    Ok(())
}

pub fn write_validation_outputs(config: &TrainConfig, summary: &ValidationSummary) -> Result<()> {
    fs::create_dir_all(&config.output_dir)?;
    let output_dir = Path::new(&config.output_dir);
    let mut metrics = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_dir.join("metrics.jsonl"))?;
    writeln!(
        metrics,
        "{}",
        serde_json::to_string(&serde_json::json!({
            "record_type": "validation",
            "metrics": summary,
        }))?
    )?;

    let summary_path = output_dir.join("summary.json");
    let mut root = if summary_path.is_file() {
        serde_json::from_str::<serde_json::Value>(&fs::read_to_string(&summary_path)?)?
    } else {
        serde_json::json!({})
    };
    match &mut root {
        serde_json::Value::Object(map) => {
            map.insert("validation".to_string(), serde_json::to_value(summary)?);
        }
        other => {
            *other = serde_json::json!({
                "previous_summary": other.clone(),
                "validation": summary,
            });
        }
    }
    fs::write(summary_path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

fn best_ious(pred: &[PixelBox], gt: &[PixelBox]) -> Vec<f32> {
    pred.iter()
        .map(|pred_box| {
            gt.iter()
                .map(|gt_box| crate::metrics::bbox_iou(*pred_box, *gt_box))
                .fold(0.0_f32, f32::max)
        })
        .collect()
}

pub fn restore_gt_boxes_to_output_space(boxes: &[PixelBox], meta: &ImageMeta) -> Vec<PixelBox> {
    boxes
        .iter()
        .copied()
        .map(|bbox| crate::preprocess::restore_box_to_output_space(bbox, meta))
        .collect()
}

fn error_record_from_sample(
    sample: &crate::dataset::DatasetSample,
    pred_boxes: Vec<PixelBox>,
    gt_boxes: Vec<PixelBox>,
    iou: Vec<f32>,
    reason: &str,
) -> ErrorRecord {
    ErrorRecord {
        image: sample.image_path.to_string_lossy().to_string(),
        source: sample.source.clone(),
        frame_id: sample.frame_id.clone(),
        width: sample.original_width,
        height: sample.original_height,
        pred_boxes,
        gt_boxes,
        iou,
        reason: reason.to_string(),
    }
}

fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> Result<()> {
    let mut text = String::new();
    for record in records {
        text.push_str(&serde_json::to_string(record)?);
        text.push('\n');
    }
    fs::write(path, text)?;
    Ok(())
}

fn average(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

pub fn loss_to_json(loss: &LossBreakdown) -> serde_json::Value {
    serde_json::to_value(loss).unwrap_or_default()
}
