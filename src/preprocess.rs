use anyhow::{Result, bail};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{config::TrainConfig, dataset::DatasetSample};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct YoloBox {
    pub class_id: i64,
    pub x_center: f32,
    pub y_center: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PixelBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RectanglePolygon {
    pub points: Vec<Point>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMeta {
    pub image_path: String,
    pub sample_id: String,
    pub original_width: u32,
    pub original_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub scale: f32,
    pub pad: [u32; 4],
    pub source: Option<String>,
    pub frame_id: Option<String>,
    pub coordinate_space: CoordinateSpace,
    pub roi_offset: Option<[i32; 2]>,
    pub frame_width: Option<u32>,
    pub frame_height: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateSpace {
    Image,
    OriginalFrame,
}

#[derive(Debug, Clone)]
pub struct PreprocessedSample {
    pub image: Vec<f32>,
    pub channels: usize,
    pub width: usize,
    pub height: usize,
    pub boxes: Vec<PixelBox>,
    pub rectangle_polygons: Vec<RectanglePolygon>,
    pub ignore_regions: Vec<PixelBox>,
    pub meta: ImageMeta,
}

#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub imgs: Vec<Vec<f32>>,
    pub gt_texts: Vec<Vec<f32>>,
    pub gt_kernels: Vec<Vec<f32>>,
    pub training_masks: Vec<Vec<f32>>,
    pub gt_instances: Vec<Vec<u32>>,
    pub gt_boxes: Vec<Vec<PixelBox>>,
    pub img_metas: Vec<ImageMeta>,
    pub width: usize,
    pub height: usize,
}

impl TrainingBatch {
    pub fn image_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 4> {
        let batch = self.imgs.len();
        let mut values = Vec::with_capacity(batch * 3 * self.height * self.width);
        for image in &self.imgs {
            values.extend_from_slice(image);
        }
        Tensor::from_data(
            TensorData::new(values, [batch, 3, self.height, self.width]),
            device,
        )
    }

    pub fn text_tensor<B: Backend>(
        &self,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        mask_tensor_downsampled(
            &self.gt_texts,
            self.height,
            self.width,
            height,
            width,
            device,
        )
    }

    pub fn kernel_tensor<B: Backend>(
        &self,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        mask_tensor_downsampled(
            &self.gt_kernels,
            self.height,
            self.width,
            height,
            width,
            device,
        )
    }

    pub fn training_mask_tensor<B: Backend>(
        &self,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        mask_tensor_downsampled(
            &self.training_masks,
            self.height,
            self.width,
            height,
            width,
            device,
        )
    }
}

impl YoloBox {
    pub fn validate(&self) -> Result<()> {
        if self.width <= 0.0 || self.height <= 0.0 {
            bail!("bbox width and height must be positive");
        }
        let values = [self.x_center, self.y_center, self.width, self.height];
        if values.iter().any(|value| !value.is_finite()) {
            bail!("bbox values must be finite");
        }
        if self.x_center < -0.25
            || self.x_center > 1.25
            || self.y_center < -0.25
            || self.y_center > 1.25
            || self.width > 1.5
            || self.height > 1.5
        {
            bail!("bbox is outside reasonable normalized range");
        }
        Ok(())
    }
}

impl PixelBox {
    pub fn from_array(values: [f32; 4]) -> Self {
        Self {
            x1: values[0],
            y1: values[1],
            x2: values[2],
            y2: values[3],
        }
        .ordered()
    }

    pub fn ordered(self) -> Self {
        Self {
            x1: self.x1.min(self.x2),
            y1: self.y1.min(self.y2),
            x2: self.x1.max(self.x2),
            y2: self.y1.max(self.y2),
        }
    }

    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn is_valid(&self) -> bool {
        self.width() > 0.5 && self.height() > 0.5
    }

    pub fn clip(self, width: u32, height: u32) -> Self {
        Self {
            x1: self.x1.clamp(0.0, width as f32),
            y1: self.y1.clamp(0.0, height as f32),
            x2: self.x2.clamp(0.0, width as f32),
            y2: self.y2.clamp(0.0, height as f32),
        }
        .ordered()
    }

    pub fn scale(self, sx: f32, sy: f32) -> Self {
        Self {
            x1: self.x1 * sx,
            y1: self.y1 * sy,
            x2: self.x2 * sx,
            y2: self.y2 * sy,
        }
    }

    pub fn translate(self, dx: f32, dy: f32) -> Self {
        Self {
            x1: self.x1 + dx,
            y1: self.y1 + dy,
            x2: self.x2 + dx,
            y2: self.y2 + dy,
        }
    }

    pub fn to_polygon(&self) -> RectanglePolygon {
        RectanglePolygon {
            points: vec![
                Point {
                    x: self.x1,
                    y: self.y1,
                },
                Point {
                    x: self.x2,
                    y: self.y1,
                },
                Point {
                    x: self.x2,
                    y: self.y2,
                },
                Point {
                    x: self.x1,
                    y: self.y2,
                },
            ],
        }
    }

    pub fn size_bucket(&self) -> String {
        let area = self.area();
        if area < 1_024.0 {
            "small".to_string()
        } else if area < 16_384.0 {
            "medium".to_string()
        } else {
            "large".to_string()
        }
    }

    pub fn vertical_bucket(&self, image_height: u32) -> String {
        let center_y = (self.y1 + self.y2) * 0.5 / image_height.max(1) as f32;
        if center_y < 0.33 {
            "top".to_string()
        } else if center_y < 0.66 {
            "middle".to_string()
        } else {
            "bottom".to_string()
        }
    }
}

impl RectanglePolygon {
    pub fn bounding_box(&self) -> PixelBox {
        let mut x1 = f32::MAX;
        let mut y1 = f32::MAX;
        let mut x2 = f32::MIN;
        let mut y2 = f32::MIN;
        for point in &self.points {
            x1 = x1.min(point.x);
            y1 = y1.min(point.y);
            x2 = x2.max(point.x);
            y2 = y2.max(point.y);
        }
        PixelBox { x1, y1, x2, y2 }
    }

    pub fn is_valid(&self) -> bool {
        self.points.len() >= 3 && self.bounding_box().is_valid()
    }

    pub fn scale(&self, sx: f32, sy: f32) -> Self {
        Self {
            points: self
                .points
                .iter()
                .map(|point| Point {
                    x: point.x * sx,
                    y: point.y * sy,
                })
                .collect(),
        }
    }

    pub fn translate(&self, dx: f32, dy: f32) -> Self {
        Self {
            points: self
                .points
                .iter()
                .map(|point| Point {
                    x: point.x + dx,
                    y: point.y + dy,
                })
                .collect(),
        }
    }

    pub fn horizontal_flip(&self, width: u32) -> Self {
        Self {
            points: self
                .points
                .iter()
                .map(|point| Point {
                    x: width as f32 - point.x,
                    y: point.y,
                })
                .collect(),
        }
    }

    pub fn rotate_about_center(&self, angle_degrees: f32, width: u32, height: u32) -> Self {
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        let angle = angle_degrees.to_radians();
        let (sin, cos) = angle.sin_cos();
        Self {
            points: self
                .points
                .iter()
                .map(|point| {
                    let x = point.x - cx;
                    let y = point.y - cy;
                    Point {
                        x: x * cos - y * sin + cx,
                        y: x * sin + y * cos + cy,
                    }
                })
                .collect(),
        }
    }

    pub fn clip_to_rect(&self, width: u32, height: u32) -> Option<Self> {
        let points = clip_polygon_to_rect(&self.points, width as f32, height as f32);
        let polygon = Self { points };
        polygon.is_valid().then_some(polygon)
    }
}

#[derive(Debug, Clone)]
struct GeometryState {
    image: RgbImage,
    boxes: Vec<PixelBox>,
    rectangle_polygons: Vec<RectanglePolygon>,
    ignore_regions: Vec<PixelBox>,
    scale: f32,
    pad: [u32; 4],
}

pub fn yolo_to_pixel(bbox: YoloBox, width: u32, height: u32) -> Result<PixelBox> {
    bbox.validate()?;
    let image_width = width as f32;
    let image_height = height as f32;
    let box_width = bbox.width * image_width;
    let box_height = bbox.height * image_height;
    let center_x = bbox.x_center * image_width;
    let center_y = bbox.y_center * image_height;
    Ok(PixelBox {
        x1: center_x - box_width * 0.5,
        y1: center_y - box_height * 0.5,
        x2: center_x + box_width * 0.5,
        y2: center_y + box_height * 0.5,
    }
    .clip(width, height))
}

pub fn scale_aligned_short(
    width: u32,
    height: u32,
    short_size: u32,
    alignment: u32,
) -> (u32, u32, f32) {
    let short = width.min(height).max(1) as f32;
    let scale = short_size as f32 / short;
    let scaled_w = (width as f32 * scale).round() as u32;
    let scaled_h = (height as f32 * scale).round() as u32;
    let aligned_w = align_up(scaled_w.max(alignment), alignment);
    let aligned_h = align_up(scaled_h.max(alignment), alignment);
    (aligned_w, aligned_h, scale)
}

pub fn restore_box_to_original_image(bbox: PixelBox, meta: &ImageMeta) -> PixelBox {
    bbox.translate(-(meta.pad[0] as f32), -(meta.pad[1] as f32))
        .scale(1.0 / meta.scale, 1.0 / meta.scale)
        .clip(meta.original_width, meta.original_height)
}

pub fn restore_box_to_output_space(bbox: PixelBox, meta: &ImageMeta) -> PixelBox {
    let restored = restore_box_to_original_image(bbox, meta);
    match (meta.coordinate_space, meta.roi_offset) {
        (CoordinateSpace::OriginalFrame, Some([dx, dy])) => {
            let frame_width = meta.frame_width.unwrap_or(meta.original_width);
            let frame_height = meta.frame_height.unwrap_or(meta.original_height);
            restored
                .translate(dx as f32, dy as f32)
                .clip(frame_width, frame_height)
        }
        _ => restored,
    }
}

pub fn preprocess_sample(
    sample: &DatasetSample,
    config: &TrainConfig,
    training: bool,
) -> Result<PreprocessedSample> {
    let image = image::ImageReader::open(&sample.image_path)?
        .decode()?
        .to_rgb8();
    let state = if training && config.augment_enabled {
        preprocess_training_state(sample, config, image)
    } else {
        preprocess_validation_state(sample, config, image)
    };
    let image_tensor = normalize_to_chw(&state.image);
    Ok(PreprocessedSample {
        image: image_tensor,
        channels: 3,
        width: state.image.width() as usize,
        height: state.image.height() as usize,
        boxes: state.boxes,
        rectangle_polygons: state.rectangle_polygons,
        ignore_regions: state.ignore_regions,
        meta: ImageMeta {
            image_path: sample.image_path.to_string_lossy().to_string(),
            sample_id: sample.sample_id.clone(),
            original_width: sample.original_width,
            original_height: sample.original_height,
            resized_width: state
                .image
                .width()
                .saturating_sub(state.pad[0] + state.pad[2]),
            resized_height: state
                .image
                .height()
                .saturating_sub(state.pad[1] + state.pad[3]),
            scale: state.scale,
            pad: state.pad,
            source: sample.source.clone(),
            frame_id: sample.frame_id.clone(),
            coordinate_space: CoordinateSpace::Image,
            roi_offset: sample
                .annotation_meta
                .as_ref()
                .and_then(|meta| meta.roi_offset),
            frame_width: sample.annotation_meta.as_ref().and_then(|meta| {
                meta.filter_region
                    .map(|region| region[2].max(0.0).round() as u32)
            }),
            frame_height: sample.annotation_meta.as_ref().and_then(|meta| {
                meta.filter_region
                    .map(|region| region[3].max(0.0).round() as u32)
            }),
        },
    })
}

pub fn with_original_frame_output(mut meta: ImageMeta) -> ImageMeta {
    meta.coordinate_space = CoordinateSpace::OriginalFrame;
    meta
}

pub fn collate_batch(samples: Vec<PreprocessedSample>) -> TrainingBatch {
    let width = samples.iter().map(|sample| sample.width).max().unwrap_or(0);
    let height = samples
        .iter()
        .map(|sample| sample.height)
        .max()
        .unwrap_or(0);
    let mut imgs = Vec::with_capacity(samples.len());
    let mut gt_texts = Vec::with_capacity(samples.len());
    let mut gt_kernels = Vec::with_capacity(samples.len());
    let mut training_masks = Vec::with_capacity(samples.len());
    let mut gt_instances = Vec::with_capacity(samples.len());
    let mut gt_boxes = Vec::with_capacity(samples.len());
    let mut img_metas = Vec::with_capacity(samples.len());
    for mut sample in samples {
        let targets = crate::target::generate_targets_from_polygons(
            sample.width,
            sample.height,
            &sample.rectangle_polygons,
            &sample.ignore_regions,
        );
        imgs.push(pad_chw_image(
            &sample.image,
            sample.channels,
            sample.width,
            sample.height,
            width,
            height,
        ));
        gt_texts.push(pad_mask(
            &targets.gt_text,
            sample.width,
            sample.height,
            width,
            height,
            0.0,
        ));
        gt_kernels.push(pad_mask(
            &targets.gt_kernel,
            sample.width,
            sample.height,
            width,
            height,
            0.0,
        ));
        training_masks.push(pad_mask(
            &targets.training_mask,
            sample.width,
            sample.height,
            width,
            height,
            1.0,
        ));
        gt_instances.push(pad_instance_mask(
            &targets.gt_instance,
            sample.width,
            sample.height,
            width,
            height,
        ));
        sample.meta.pad[2] += (width - sample.width) as u32;
        sample.meta.pad[3] += (height - sample.height) as u32;
        gt_boxes.push(sample.boxes);
        img_metas.push(sample.meta);
    }
    TrainingBatch {
        imgs,
        gt_texts,
        gt_kernels,
        training_masks,
        gt_instances,
        gt_boxes,
        img_metas,
        width,
        height,
    }
}

pub fn collate_batch_with_config(
    samples: Vec<PreprocessedSample>,
    config: &TrainConfig,
) -> TrainingBatch {
    let width = samples.iter().map(|sample| sample.width).max().unwrap_or(0);
    let height = samples
        .iter()
        .map(|sample| sample.height)
        .max()
        .unwrap_or(0);
    let mut imgs = Vec::with_capacity(samples.len());
    let mut gt_texts = Vec::with_capacity(samples.len());
    let mut gt_kernels = Vec::with_capacity(samples.len());
    let mut training_masks = Vec::with_capacity(samples.len());
    let mut gt_instances = Vec::with_capacity(samples.len());
    let mut gt_boxes = Vec::with_capacity(samples.len());
    let mut img_metas = Vec::with_capacity(samples.len());
    for mut sample in samples {
        let targets = crate::target::generate_targets_from_polygons_with_config(
            sample.width,
            sample.height,
            &sample.rectangle_polygons,
            &sample.ignore_regions,
            crate::target::TargetConfig {
                pooling_size: config.pooling_size,
                shrink_kernel_scale: config.shrink_kernel_scale,
                min_kernel_width: config.min_kernel_width,
                min_kernel_height: config.min_kernel_height,
            },
        );
        imgs.push(pad_chw_image(
            &sample.image,
            sample.channels,
            sample.width,
            sample.height,
            width,
            height,
        ));
        gt_texts.push(pad_mask(
            &targets.gt_text,
            sample.width,
            sample.height,
            width,
            height,
            0.0,
        ));
        gt_kernels.push(pad_mask(
            &targets.gt_kernel,
            sample.width,
            sample.height,
            width,
            height,
            0.0,
        ));
        training_masks.push(pad_mask(
            &targets.training_mask,
            sample.width,
            sample.height,
            width,
            height,
            1.0,
        ));
        gt_instances.push(pad_instance_mask(
            &targets.gt_instance,
            sample.width,
            sample.height,
            width,
            height,
        ));
        sample.meta.pad[2] += (width - sample.width) as u32;
        sample.meta.pad[3] += (height - sample.height) as u32;
        gt_boxes.push(sample.boxes);
        img_metas.push(sample.meta);
    }
    TrainingBatch {
        imgs,
        gt_texts,
        gt_kernels,
        training_masks,
        gt_instances,
        gt_boxes,
        img_metas,
        width,
        height,
    }
}

fn pad_chw_image(
    image: &[f32],
    channels: usize,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    if src_w == dst_w && src_h == dst_h {
        return image.to_vec();
    }
    let mut padded = vec![0.0; channels * dst_w * dst_h];
    for channel in 0..channels {
        for y in 0..src_h {
            let src_offset = channel * src_w * src_h + y * src_w;
            let dst_offset = channel * dst_w * dst_h + y * dst_w;
            padded[dst_offset..dst_offset + src_w]
                .copy_from_slice(&image[src_offset..src_offset + src_w]);
        }
    }
    padded
}

fn pad_mask(
    mask: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    pad_value: f32,
) -> Vec<f32> {
    if src_w == dst_w && src_h == dst_h {
        return mask.to_vec();
    }
    let mut padded = vec![pad_value; dst_w * dst_h];
    for y in 0..src_h {
        let src_offset = y * src_w;
        let dst_offset = y * dst_w;
        padded[dst_offset..dst_offset + src_w]
            .copy_from_slice(&mask[src_offset..src_offset + src_w]);
    }
    padded
}

fn pad_instance_mask(
    mask: &[u32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u32> {
    if src_w == dst_w && src_h == dst_h {
        return mask.to_vec();
    }
    let mut padded = vec![0; dst_w * dst_h];
    for y in 0..src_h {
        let src_offset = y * src_w;
        let dst_offset = y * dst_w;
        padded[dst_offset..dst_offset + src_w]
            .copy_from_slice(&mask[src_offset..src_offset + src_w]);
    }
    padded
}

fn preprocess_validation_state(
    sample: &DatasetSample,
    config: &TrainConfig,
    image: RgbImage,
) -> GeometryState {
    let (target_w, target_h, scale) = scale_aligned_short(
        sample.original_width,
        sample.original_height,
        config.short_size,
        config.alignment,
    );
    let resized_w = ((sample.original_width as f32) * scale).round().max(1.0) as u32;
    let resized_h = ((sample.original_height as f32) * scale).round().max(1.0) as u32;
    let resized = DynamicImage::ImageRgb8(image)
        .resize_exact(resized_w, resized_h, FilterType::Triangle)
        .to_rgb8();
    let mut canvas = RgbImage::new(target_w, target_h);
    image::imageops::replace(&mut canvas, &resized, 0, 0);
    let scaled_polygons =
        scale_clip_polygons(&sample.rectangle_polygons, scale, scale, target_w, target_h);
    let scaled_ignore_regions =
        scale_clip_boxes(&sample.ignore_regions, scale, scale, target_w, target_h);
    GeometryState {
        image: canvas,
        boxes: polygons_to_boxes(&scaled_polygons),
        rectangle_polygons: scaled_polygons,
        ignore_regions: scaled_ignore_regions,
        scale,
        pad: [
            0,
            0,
            target_w.saturating_sub(resized_w),
            target_h.saturating_sub(resized_h),
        ],
    }
}

fn preprocess_training_state(
    sample: &DatasetSample,
    config: &TrainConfig,
    image: RgbImage,
) -> GeometryState {
    let mut rng = StdRng::seed_from_u64(config.seed ^ stable_hash(&sample.sample_id));
    let mut state = GeometryState {
        image,
        boxes: sample.pixel_boxes_after_label_masks.clone(),
        rectangle_polygons: sample.rectangle_polygons.clone(),
        ignore_regions: sample.ignore_regions.clone(),
        scale: 1.0,
        pad: [0, 0, 0, 0],
    };
    if config.augment_enabled {
        state = fast_random_scale(state, config, &mut rng);
        if config.random_horizontal_flip && rng.random::<f32>() < config.flip_prob {
            state = fast_random_horizontal_flip(state);
        }
        if config.random_rotate && config.rotate_angle > 0.0 {
            let angle = rng.random_range(-config.rotate_angle..=config.rotate_angle);
            state = fast_random_rotate(state, angle);
        }
    }
    state = fast_random_crop_padding(state, config.input_width(), config.input_height(), &mut rng);
    if config.augment_enabled {
        apply_color_jitter_and_blur(&mut state.image, config, &mut rng);
    }
    state
}

fn fast_random_scale(
    mut state: GeometryState,
    config: &TrainConfig,
    rng: &mut StdRng,
) -> GeometryState {
    let scale = rng.random_range(config.scale_min..=config.scale_max);
    let aspect = rng.random_range(config.aspect_min..=config.aspect_max);
    let sx = scale * aspect.sqrt();
    let sy = scale / aspect.sqrt();
    let new_w = ((state.image.width() as f32) * sx).round().max(1.0) as u32;
    let new_h = ((state.image.height() as f32) * sy).round().max(1.0) as u32;
    state.image = DynamicImage::ImageRgb8(state.image)
        .resize_exact(new_w, new_h, FilterType::Triangle)
        .to_rgb8();
    state.rectangle_polygons = scale_clip_polygons(&state.rectangle_polygons, sx, sy, new_w, new_h);
    state.boxes = polygons_to_boxes(&state.rectangle_polygons);
    state.ignore_regions = scale_clip_boxes(&state.ignore_regions, sx, sy, new_w, new_h);
    state.scale *= sx.min(sy);
    state
}

fn fast_random_horizontal_flip(mut state: GeometryState) -> GeometryState {
    let width = state.image.width();
    image::imageops::flip_horizontal_in_place(&mut state.image);
    state.boxes = state
        .rectangle_polygons
        .iter()
        .map(|polygon| polygon.horizontal_flip(width))
        .filter_map(|polygon| polygon.clip_to_rect(width, state.image.height()))
        .map(|polygon| polygon.bounding_box())
        .collect();
    state.rectangle_polygons = state
        .rectangle_polygons
        .iter()
        .map(|polygon| polygon.horizontal_flip(width))
        .filter_map(|polygon| polygon.clip_to_rect(width, state.image.height()))
        .collect();
    state.ignore_regions = state
        .ignore_regions
        .iter()
        .map(|bbox| flip_box(*bbox, width))
        .collect();
    state
}

fn fast_random_rotate(mut state: GeometryState, angle_degrees: f32) -> GeometryState {
    let width = state.image.width();
    let height = state.image.height();
    state.image = rotate_about_center(
        &state.image,
        angle_degrees.to_radians(),
        Interpolation::Bilinear,
        Rgb([0, 0, 0]),
    );
    state.rectangle_polygons = state
        .rectangle_polygons
        .iter()
        .map(|polygon| polygon.rotate_about_center(angle_degrees, width, height))
        .filter_map(|polygon| polygon.clip_to_rect(width, height))
        .collect();
    state.boxes = polygons_to_boxes(&state.rectangle_polygons);
    state.ignore_regions = state
        .ignore_regions
        .iter()
        .map(|bbox| rotate_box(*bbox, angle_degrees, width, height))
        .filter(PixelBox::is_valid)
        .collect();
    state
}

fn fast_random_crop_padding(
    mut state: GeometryState,
    target_w: u32,
    target_h: u32,
    rng: &mut StdRng,
) -> GeometryState {
    let src_w = state.image.width();
    let src_h = state.image.height();
    let crop_w = target_w.min(src_w);
    let crop_h = target_h.min(src_h);
    let (crop_x, crop_y) = choose_crop_origin(&state.boxes, src_w, src_h, crop_w, crop_h, rng);
    let cropped =
        image::imageops::crop_imm(&state.image, crop_x, crop_y, crop_w, crop_h).to_image();
    let mut canvas = RgbImage::new(target_w, target_h);
    image::imageops::replace(&mut canvas, &cropped, 0, 0);
    state.image = canvas;
    state.rectangle_polygons = crop_polygons(
        &state.rectangle_polygons,
        crop_x,
        crop_y,
        target_w,
        target_h,
    );
    state.boxes = polygons_to_boxes(&state.rectangle_polygons);
    state.ignore_regions = crop_boxes(&state.ignore_regions, crop_x, crop_y, target_w, target_h);
    state.pad = [
        0,
        0,
        target_w.saturating_sub(crop_w),
        target_h.saturating_sub(crop_h),
    ];
    state
}

fn choose_crop_origin(
    boxes: &[PixelBox],
    src_w: u32,
    src_h: u32,
    crop_w: u32,
    crop_h: u32,
    rng: &mut StdRng,
) -> (u32, u32) {
    if let Some(bbox) = boxes.get(rng.random_range(0..boxes.len().max(1))) {
        let cx = ((bbox.x1 + bbox.x2) * 0.5).round().max(0.0) as u32;
        let cy = ((bbox.y1 + bbox.y2) * 0.5).round().max(0.0) as u32;
        let x = cx
            .saturating_sub(crop_w / 2)
            .min(src_w.saturating_sub(crop_w));
        let y = cy
            .saturating_sub(crop_h / 2)
            .min(src_h.saturating_sub(crop_h));
        (x, y)
    } else {
        (
            rng.random_range(0..=src_w.saturating_sub(crop_w)),
            rng.random_range(0..=src_h.saturating_sub(crop_h)),
        )
    }
}

fn apply_color_jitter_and_blur(image: &mut RgbImage, config: &TrainConfig, rng: &mut StdRng) {
    let brightness = rng.random_range((1.0 - config.brightness)..=(1.0 + config.brightness));
    let contrast = rng.random_range((1.0 - config.contrast)..=(1.0 + config.contrast));
    let saturation = rng.random_range((1.0 - config.saturation)..=(1.0 + config.saturation));
    let hue = rng.random_range(-config.hue..=config.hue);
    for pixel in image.pixels_mut() {
        let mut rgb = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32];
        for channel in &mut rgb {
            *channel = ((*channel - 128.0) * contrast + 128.0) * brightness;
        }
        let gray = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114;
        for channel in &mut rgb {
            *channel = gray + (*channel - gray) * saturation;
        }
        if config.hue > 0.0 {
            rgb = rotate_hue(rgb, hue);
        }
        *pixel = Rgb([
            rgb[0].clamp(0.0, 255.0) as u8,
            rgb[1].clamp(0.0, 255.0) as u8,
            rgb[2].clamp(0.0, 255.0) as u8,
        ]);
    }
    if config.gaussian_blur && rng.random::<f32>() < config.gaussian_blur_prob {
        *image = image::imageops::blur(image, rng.random_range(0.3..=1.0));
    }
}

fn rotate_hue(rgb: [f32; 3], hue: f32) -> [f32; 3] {
    let (mut h, s, v) = rgb_to_hsv(rgb);
    h = (h + hue).rem_euclid(1.0);
    hsv_to_rgb(h, s, v)
}

fn rgb_to_hsv(rgb: [f32; 3]) -> (f32, f32, f32) {
    let r = rgb[0].clamp(0.0, 255.0) / 255.0;
    let g = rgb[1].clamp(0.0, 255.0) / 255.0;
    let b = rgb[2].clamp(0.0, 255.0) / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let hue = if delta <= f32::EPSILON {
        0.0
    } else if (max - r).abs() <= f32::EPSILON {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if (max - g).abs() <= f32::EPSILON {
        (((b - r) / delta) + 2.0) / 6.0
    } else {
        (((r - g) / delta) + 4.0) / 6.0
    };
    let saturation = if max <= f32::EPSILON {
        0.0
    } else {
        delta / max
    };
    (hue, saturation, max)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    if s <= f32::EPSILON {
        let value = v * 255.0;
        return [value, value, value];
    }
    let sector = h.rem_euclid(1.0) * 6.0;
    let i = sector.floor();
    let f = sector - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i as u32 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r * 255.0, g * 255.0, b * 255.0]
}

fn scale_clip_boxes(
    boxes: &[PixelBox],
    sx: f32,
    sy: f32,
    width: u32,
    height: u32,
) -> Vec<PixelBox> {
    boxes
        .iter()
        .map(|bbox| bbox.scale(sx, sy).clip(width, height))
        .filter(PixelBox::is_valid)
        .collect()
}

fn scale_clip_polygons(
    polygons: &[RectanglePolygon],
    sx: f32,
    sy: f32,
    width: u32,
    height: u32,
) -> Vec<RectanglePolygon> {
    polygons
        .iter()
        .filter_map(|polygon| polygon.scale(sx, sy).clip_to_rect(width, height))
        .collect()
}

fn polygons_to_boxes(polygons: &[RectanglePolygon]) -> Vec<PixelBox> {
    polygons
        .iter()
        .map(RectanglePolygon::bounding_box)
        .filter(PixelBox::is_valid)
        .collect()
}

fn crop_boxes(
    boxes: &[PixelBox],
    crop_x: u32,
    crop_y: u32,
    width: u32,
    height: u32,
) -> Vec<PixelBox> {
    boxes
        .iter()
        .map(|bbox| {
            bbox.translate(-(crop_x as f32), -(crop_y as f32))
                .clip(width, height)
        })
        .filter(PixelBox::is_valid)
        .collect()
}

fn crop_polygons(
    polygons: &[RectanglePolygon],
    crop_x: u32,
    crop_y: u32,
    width: u32,
    height: u32,
) -> Vec<RectanglePolygon> {
    polygons
        .iter()
        .filter_map(|polygon| {
            polygon
                .translate(-(crop_x as f32), -(crop_y as f32))
                .clip_to_rect(width, height)
        })
        .collect()
}

fn flip_box(bbox: PixelBox, width: u32) -> PixelBox {
    PixelBox {
        x1: width as f32 - bbox.x2,
        y1: bbox.y1,
        x2: width as f32 - bbox.x1,
        y2: bbox.y2,
    }
    .ordered()
}

fn rotate_box(bbox: PixelBox, angle_degrees: f32, width: u32, height: u32) -> PixelBox {
    random_rotate_box_for_test(bbox, angle_degrees, width, height)
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn normalize_to_chw(image: &RgbImage) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let plane = (width * height) as usize;
    let mut data = vec![0.0; plane * 3];
    let mean = [0.485_f32, 0.456, 0.406];
    let std = [0.229_f32, 0.224, 0.225];
    for (x, y, pixel) in image.enumerate_pixels() {
        let offset = (y * width + x) as usize;
        for channel in 0..3 {
            let value = pixel[channel] as f32 / 255.0;
            data[channel * plane + offset] = (value - mean[channel]) / std[channel];
        }
    }
    data
}

fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

#[derive(Debug, Clone, Copy)]
enum ClipEdge {
    Left(f32),
    Right(f32),
    Top(f32),
    Bottom(f32),
}

fn clip_polygon_to_rect(points: &[Point], width: f32, height: f32) -> Vec<Point> {
    let mut clipped = points.to_vec();
    for edge in [
        ClipEdge::Left(0.0),
        ClipEdge::Right(width),
        ClipEdge::Top(0.0),
        ClipEdge::Bottom(height),
    ] {
        clipped = clip_polygon_edge(&clipped, edge);
        if clipped.is_empty() {
            break;
        }
    }
    clipped
}

fn clip_polygon_edge(points: &[Point], edge: ClipEdge) -> Vec<Point> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut output = Vec::new();
    let mut previous = points[points.len() - 1];
    let mut previous_inside = point_inside_clip_edge(previous, edge);
    for current in points {
        let current_inside = point_inside_clip_edge(*current, edge);
        if current_inside {
            if !previous_inside {
                output.push(intersect_clip_edge(previous, *current, edge));
            }
            output.push(*current);
        } else if previous_inside {
            output.push(intersect_clip_edge(previous, *current, edge));
        }
        previous = *current;
        previous_inside = current_inside;
    }
    output
}

fn point_inside_clip_edge(point: Point, edge: ClipEdge) -> bool {
    match edge {
        ClipEdge::Left(x) => point.x >= x,
        ClipEdge::Right(x) => point.x <= x,
        ClipEdge::Top(y) => point.y >= y,
        ClipEdge::Bottom(y) => point.y <= y,
    }
}

fn intersect_clip_edge(start: Point, end: Point, edge: ClipEdge) -> Point {
    match edge {
        ClipEdge::Left(x) | ClipEdge::Right(x) => {
            let denom = end.x - start.x;
            if denom.abs() <= f32::EPSILON {
                return Point { x, y: start.y };
            }
            let t = ((x - start.x) / denom).clamp(0.0, 1.0);
            Point {
                x,
                y: start.y + (end.y - start.y) * t,
            }
        }
        ClipEdge::Top(y) | ClipEdge::Bottom(y) => {
            let denom = end.y - start.y;
            if denom.abs() <= f32::EPSILON {
                return Point { x: start.x, y };
            }
            let t = ((y - start.y) / denom).clamp(0.0, 1.0);
            Point {
                x: start.x + (end.x - start.x) * t,
                y,
            }
        }
    }
}

fn mask_tensor<B: Backend>(
    masks: &[Vec<f32>],
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let batch = masks.len();
    let mut values = Vec::with_capacity(batch * height * width);
    for mask in masks {
        values.extend_from_slice(mask);
    }
    Tensor::from_data(TensorData::new(values, [batch, 1, height, width]), device)
}

fn mask_tensor_downsampled<B: Backend>(
    masks: &[Vec<f32>],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    if src_h == dst_h && src_w == dst_w {
        return mask_tensor(masks, src_h, src_w, device);
    }
    let batch = masks.len();
    let mut values = Vec::with_capacity(batch * dst_h * dst_w);
    for mask in masks {
        for y in 0..dst_h {
            let src_y = (y * src_h / dst_h).min(src_h - 1);
            for x in 0..dst_w {
                let src_x = (x * src_w / dst_w).min(src_w - 1);
                values.push(mask[src_y * src_w + src_x]);
            }
        }
    }
    Tensor::from_data(TensorData::new(values, [batch, 1, dst_h, dst_w]), device)
}

pub fn random_scale_box_for_test(bbox: PixelBox, sx: f32, sy: f32) -> PixelBox {
    bbox.scale(sx, sy)
}

pub fn random_rotate_box_for_test(
    bbox: PixelBox,
    angle_degrees: f32,
    width: u32,
    height: u32,
) -> PixelBox {
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    let angle = angle_degrees.to_radians();
    let (sin, cos) = angle.sin_cos();
    let points = bbox
        .to_polygon()
        .points
        .into_iter()
        .map(|mut point| {
            let x = point.x - cx;
            let y = point.y - cy;
            point.x = x * cos - y * sin + cx;
            point.y = x * sin + y * cos + cy;
            point
        })
        .collect();
    RectanglePolygon { points }
        .bounding_box()
        .clip(width, height)
}

pub fn random_scale_polygon_for_test(
    polygon: RectanglePolygon,
    sx: f32,
    sy: f32,
) -> RectanglePolygon {
    polygon.scale(sx, sy)
}

pub fn random_rotate_polygon_for_test(
    polygon: RectanglePolygon,
    angle_degrees: f32,
    width: u32,
    height: u32,
) -> RectanglePolygon {
    polygon
        .rotate_about_center(angle_degrees, width, height)
        .clip_to_rect(width, height)
        .expect("rotated test polygon should remain valid")
}

pub fn crop_padding_polygon_for_test(
    polygon: RectanglePolygon,
    crop_x: u32,
    crop_y: u32,
    width: u32,
    height: u32,
) -> Option<RectanglePolygon> {
    polygon
        .translate(-(crop_x as f32), -(crop_y as f32))
        .clip_to_rect(width, height)
}

pub fn crop_padding_box_for_test(
    bbox: PixelBox,
    crop_x: u32,
    crop_y: u32,
    width: u32,
    height: u32,
) -> Option<PixelBox> {
    let cropped = bbox
        .translate(-(crop_x as f32), -(crop_y as f32))
        .clip(width, height);
    cropped.is_valid().then_some(cropped)
}

pub fn hue_jitter_rgb_for_test(rgb: [u8; 3], hue: f32) -> [u8; 3] {
    let shifted = rotate_hue([rgb[0] as f32, rgb[1] as f32, rgb[2] as f32], hue);
    [
        shifted[0].clamp(0.0, 255.0) as u8,
        shifted[1].clamp(0.0, 255.0) as u8,
        shifted[2].clamp(0.0, 255.0) as u8,
    ]
}
