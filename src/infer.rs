use std::{fs, path::Path};

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use image::GenericImageView;
use serde::Serialize;

use crate::{
    checkpoint::load_checkpoint_model,
    config::TrainConfig,
    dataset::DatasetSample,
    postprocess::{PostprocessConfig, postprocess_output},
    preprocess::{
        CoordinateSpace, ImageMeta, PixelBox, PreprocessedSample, collate_batch, preprocess_sample,
        with_original_frame_output,
    },
    validate::feature_metas,
};

#[derive(Debug, Clone, Serialize)]
pub struct InferenceOutput {
    pub image: String,
    pub width: u32,
    pub height: u32,
    pub boxes: Vec<crate::model::DetectionBox>,
    pub meta: serde_json::Value,
}

pub fn infer_image(
    config: &TrainConfig,
    checkpoint: &str,
    image_path: &str,
) -> Result<InferenceOutput> {
    crate::backend::infer_with_configured_backend(config, checkpoint, image_path)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct InferRoi {
    pub offset: [i32; 2],
    pub frame_size: Option<[u32; 2]>,
}

pub fn infer_image_backend<B: Backend>(
    config: &TrainConfig,
    checkpoint: &str,
    image_path: &str,
) -> Result<InferenceOutput> {
    infer_image_backend_with_roi::<B>(config, checkpoint, image_path, None)
}

pub fn infer_image_backend_with_roi<B: Backend>(
    config: &TrainConfig,
    checkpoint: &str,
    image_path: &str,
    roi: Option<InferRoi>,
) -> Result<InferenceOutput> {
    let device = B::Device::default();
    let (model, _checkpoint_meta) = load_checkpoint_model::<B>(checkpoint, &device)?;
    let image = image::ImageReader::open(image_path)
        .with_context(|| format!("failed to open image {image_path}"))?
        .decode()?;
    let (width, height) = image.dimensions();
    let sample = DatasetSample {
        image_path: Path::new(image_path).to_path_buf(),
        label_path: Path::new("").to_path_buf(),
        root_id: 0,
        sample_id: Path::new(image_path)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("image")
            .to_string(),
        original_width: width,
        original_height: height,
        yolo_boxes_raw: Vec::new(),
        pixel_boxes_raw: Vec::new(),
        pixel_boxes_after_label_masks: Vec::new(),
        rectangle_polygons: Vec::new(),
        ignore_regions: Vec::new(),
        annotation_meta: None,
        source: None,
        frame_id: None,
        issues: Vec::new(),
        ignored: false,
        ignore_reason: None,
    };
    let preprocessed = preprocess_sample(&sample, config, false)?;
    let batch = collate_infer(preprocessed, roi);
    let output = crate::model::output_to_cpu(model.forward(batch.image_tensor(&device)));
    let metas = feature_metas(
        &batch.img_metas,
        batch.width,
        batch.height,
        output.width,
        output.height,
    );
    let boxes = postprocess_output(
        &output,
        &metas,
        &PostprocessConfig {
            threshold_region: config.threshold_region,
            threshold_kernel: config.threshold_kernel,
            min_width: config.min_kernel_width as f32,
            min_height: config.min_kernel_height as f32,
            max_width_ratio: config.max_detection_width_ratio,
        },
    )
    .into_iter()
    .next()
    .unwrap_or_default();
    Ok(InferenceOutput {
        image: image_path.to_string(),
        width: roi
            .and_then(|roi| roi.frame_size.map(|size| size[0]))
            .unwrap_or(width),
        height: roi
            .and_then(|roi| roi.frame_size.map(|size| size[1]))
            .unwrap_or(height),
        boxes,
        meta: serde_json::json!({
            "source": null,
            "frame_id": null,
            "coordinate_space": if roi.is_some() { "original_frame" } else { "image" },
            "roi_offset": roi.map(|roi| roi.offset),
            "roi_image_size": if roi.is_some() { Some([width, height]) } else { None },
        }),
    })
}

fn collate_infer(
    sample: PreprocessedSample,
    roi: Option<InferRoi>,
) -> crate::preprocess::TrainingBatch {
    let meta = if let Some(roi) = roi {
        with_original_frame_output(ImageMeta {
            roi_offset: Some(roi.offset),
            frame_width: roi.frame_size.map(|size| size[0]),
            frame_height: roi.frame_size.map(|size| size[1]),
            coordinate_space: CoordinateSpace::OriginalFrame,
            ..sample.meta
        })
    } else {
        sample.meta.clone()
    };
    collate_batch(vec![PreprocessedSample {
        boxes: Vec::<PixelBox>::new(),
        rectangle_polygons: Vec::new(),
        ignore_regions: Vec::<PixelBox>::new(),
        meta,
        ..sample
    }])
}

pub fn write_inference_json(output: &InferenceOutput) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(output)?);
    Ok(())
}

pub fn save_inference_json(path: impl AsRef<Path>, output: &InferenceOutput) -> Result<()> {
    fs::write(path, serde_json::to_string_pretty(output)?)?;
    Ok(())
}
