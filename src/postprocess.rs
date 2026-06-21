use std::{collections::VecDeque, time::Instant};

use crate::{
    model::{CpuModelOutput, DetectionBox, sigmoid},
    preprocess::{ImageMeta, PixelBox, restore_box_to_output_space},
};

#[derive(Debug, Clone)]
pub struct PostprocessConfig {
    pub threshold_region: f32,
    pub threshold_kernel: f32,
    pub min_width: f32,
    pub min_height: f32,
    pub max_width_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct PostprocessImageResult {
    pub boxes: Vec<DetectionBox>,
    pub postprocess_latency_ms: f32,
    pub candidate_count: usize,
    pub final_box_count: usize,
}

pub fn postprocess_output(
    output: &CpuModelOutput,
    metas: &[ImageMeta],
    config: &PostprocessConfig,
) -> Vec<Vec<DetectionBox>> {
    postprocess_output_with_stats(output, metas, config)
        .into_iter()
        .map(|result| result.boxes)
        .collect()
}

pub fn postprocess_output_with_stats(
    output: &CpuModelOutput,
    metas: &[ImageMeta],
    config: &PostprocessConfig,
) -> Vec<PostprocessImageResult> {
    output
        .text_region_logits
        .iter()
        .zip(&output.kernel_logits)
        .zip(metas)
        .map(|((region, kernel), meta)| {
            let start = Instant::now();
            let (boxes, candidate_count) =
                components_to_boxes(region, kernel, output.width, output.height, meta, config);
            let final_box_count = boxes.len();
            PostprocessImageResult {
                boxes,
                postprocess_latency_ms: start.elapsed().as_secs_f32() * 1000.0,
                candidate_count,
                final_box_count,
            }
        })
        .collect()
}

fn components_to_boxes(
    region: &[f32],
    kernel: &[f32],
    width: usize,
    height: usize,
    meta: &ImageMeta,
    config: &PostprocessConfig,
) -> (Vec<DetectionBox>, usize) {
    let plane = width * height;
    let region_mask = region
        .iter()
        .map(|value| sigmoid(*value) >= config.threshold_region)
        .collect::<Vec<_>>();
    let kernel_mask = kernel
        .iter()
        .enumerate()
        .map(|(index, value)| region_mask[index] && sigmoid(*value) >= config.threshold_kernel)
        .collect::<Vec<_>>();
    let seeds = kernel_components(&kernel_mask, width, height);
    let candidate_count = seeds.len();
    let mut owner = vec![usize::MAX; plane];
    let mut queue = VecDeque::new();
    for (seed_id, pixels) in seeds.iter().enumerate() {
        for &(x, y) in pixels {
            let index = y * width + x;
            owner[index] = seed_id;
            queue.push_back((x, y, seed_id));
        }
    }
    while let Some((cx, cy, seed_id)) = queue.pop_front() {
        for (nx, ny) in neighbors(cx, cy, width, height) {
            let next = ny * width + nx;
            if region_mask[next] && owner[next] == usize::MAX {
                owner[next] = seed_id;
                queue.push_back((nx, ny, seed_id));
            }
        }
    }
    let mut stats = (0..seeds.len())
        .map(|seed_id| ComponentStats::new(seed_id, width, height))
        .collect::<Vec<_>>();
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let seed_id = owner[index];
            if seed_id == usize::MAX {
                continue;
            }
            stats[seed_id].add(x, y, sigmoid(region[index]), sigmoid(kernel[index]));
        }
    }
    let mut boxes = Vec::new();
    for stat in stats {
        if stat.count == 0 {
            continue;
        }
        let resized_box = clamp_component_width(
            refined_component_box(&owner, stat.seed_id, region, width, height, &stat),
            &stat,
            width,
            config.max_width_ratio,
        );
        let original = restore_box_to_output_space(resized_box, meta);
        if original.width() >= config.min_width && original.height() >= config.min_height {
            boxes.push(DetectionBox::from((original, stat.confidence())));
        }
    }
    (boxes, candidate_count)
}

fn kernel_components(mask: &[bool], width: usize, height: usize) -> Vec<Vec<(usize, usize)>> {
    let mut visited = vec![false; width * height];
    let mut components = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if visited[index] || !mask[index] {
                continue;
            }
            let mut queue = VecDeque::from([(x, y)]);
            visited[index] = true;
            let mut pixels = Vec::new();
            while let Some((cx, cy)) = queue.pop_front() {
                pixels.push((cx, cy));
                for (nx, ny) in neighbors(cx, cy, width, height) {
                    let next = ny * width + nx;
                    if !visited[next] && mask[next] {
                        visited[next] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
            components.push(pixels);
        }
    }
    components
}

#[derive(Debug, Clone)]
struct ComponentStats {
    seed_id: usize,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    confidence_sum: f32,
    kernel_weight_sum: f32,
    kernel_x_sum: f32,
    count: usize,
}

impl ComponentStats {
    fn new(seed_id: usize, width: usize, height: usize) -> Self {
        Self {
            seed_id,
            min_x: width,
            min_y: height,
            max_x: 0,
            max_y: 0,
            confidence_sum: 0.0,
            kernel_weight_sum: 0.0,
            kernel_x_sum: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, x: usize, y: usize, region_prob: f32, kernel_prob: f32) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.confidence_sum += region_prob * 0.8 + kernel_prob * 0.2;
        self.kernel_weight_sum += kernel_prob;
        self.kernel_x_sum += kernel_prob * (x as f32 + 0.5);
        self.count += 1;
    }

    fn confidence(&self) -> f32 {
        self.confidence_sum / self.count.max(1) as f32
    }

    fn kernel_center_x(&self) -> Option<f32> {
        (self.kernel_weight_sum > 0.0).then_some(self.kernel_x_sum / self.kernel_weight_sum)
    }
}

fn clamp_component_width(
    bbox: PixelBox,
    stat: &ComponentStats,
    output_width: usize,
    max_width_ratio: f32,
) -> PixelBox {
    if max_width_ratio <= 0.0 || max_width_ratio >= 1.0 || output_width == 0 {
        return bbox;
    }
    let max_width = output_width as f32 * max_width_ratio;
    if bbox.width() <= max_width {
        return bbox;
    }
    let center = stat
        .kernel_center_x()
        .unwrap_or((bbox.x1 + bbox.x2) * 0.5)
        .clamp(0.0, output_width as f32);
    let mut x1 = center - max_width * 0.5;
    let mut x2 = center + max_width * 0.5;
    if x1 < 0.0 {
        x2 -= x1;
        x1 = 0.0;
    }
    if x2 > output_width as f32 {
        let overflow = x2 - output_width as f32;
        x1 = (x1 - overflow).max(0.0);
        x2 = output_width as f32;
    }
    PixelBox { x1, x2, ..bbox }
}

fn refined_component_box(
    owner: &[usize],
    seed_id: usize,
    region: &[f32],
    width: usize,
    height: usize,
    fallback: &ComponentStats,
) -> PixelBox {
    let mut max_prob = 0.0_f32;
    for (index, owner_id) in owner.iter().enumerate() {
        if *owner_id == seed_id {
            max_prob = max_prob.max(sigmoid(region[index]));
        }
    }
    let threshold = (max_prob * 0.95).max(0.5);
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut count = 0;
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if owner[index] == seed_id && sigmoid(region[index]) >= threshold {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                count += 1;
            }
        }
    }
    if count == 0 {
        return PixelBox {
            x1: fallback.min_x as f32,
            y1: fallback.min_y as f32,
            x2: (fallback.max_x + 1) as f32,
            y2: (fallback.max_y + 1) as f32,
        };
    }
    PixelBox {
        x1: min_x as f32,
        y1: min_y as f32,
        x2: (max_x + 1) as f32,
        y2: (max_y + 1) as f32,
    }
}

fn neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut values = Vec::with_capacity(4);
    if x > 0 {
        values.push((x - 1, y));
    }
    if y > 0 {
        values.push((x, y - 1));
    }
    if x + 1 < width {
        values.push((x + 1, y));
    }
    if y + 1 < height {
        values.push((x, y + 1));
    }
    values.into_iter()
}

#[cfg(test)]
mod tests {
    use super::{PostprocessConfig, postprocess_output};
    use crate::{
        model::CpuModelOutput,
        preprocess::{CoordinateSpace, ImageMeta},
    };

    #[test]
    fn trims_wide_region_component_to_high_confidence_subtitle_band() {
        let width = 16;
        let height = 4;
        let mut region = vec![-8.0; width * height];
        let mut kernel = vec![-8.0; width * height];
        for x in 0..width {
            region[width + x] = 1.0;
        }
        for x in 5..11 {
            region[width + x] = 4.0;
        }
        for x in 7..9 {
            kernel[width + x] = 4.0;
        }

        let output = CpuModelOutput {
            text_region_logits: vec![region],
            kernel_logits: vec![kernel],
            width,
            height,
        };
        let meta = ImageMeta {
            image_path: "synthetic.jpg".to_string(),
            sample_id: "synthetic".to_string(),
            original_width: width as u32,
            original_height: height as u32,
            resized_width: width as u32,
            resized_height: height as u32,
            scale: 1.0,
            pad: [0, 0, 0, 0],
            source: None,
            frame_id: None,
            coordinate_space: CoordinateSpace::Image,
            roi_offset: None,
            frame_width: None,
            frame_height: None,
        };

        let boxes = postprocess_output(
            &output,
            &[meta],
            &PostprocessConfig {
                threshold_region: 0.5,
                threshold_kernel: 0.5,
                min_width: 1.0,
                min_height: 1.0,
                max_width_ratio: 1.0,
            },
        );

        assert_eq!(boxes[0].len(), 1);
        assert_eq!(boxes[0][0].x1, 5.0);
        assert_eq!(boxes[0][0].x2, 11.0);
    }

    #[test]
    fn clamps_overwide_component_around_kernel_center_when_configured() {
        let width = 20;
        let height = 4;
        let mut region = vec![-8.0; width * height];
        let mut kernel = vec![-8.0; width * height];
        for x in 0..width {
            region[width + x] = 4.0;
        }
        for x in 9..11 {
            kernel[width + x] = 4.0;
        }

        let output = CpuModelOutput {
            text_region_logits: vec![region],
            kernel_logits: vec![kernel],
            width,
            height,
        };
        let meta = ImageMeta {
            image_path: "synthetic.jpg".to_string(),
            sample_id: "synthetic".to_string(),
            original_width: width as u32,
            original_height: height as u32,
            resized_width: width as u32,
            resized_height: height as u32,
            scale: 1.0,
            pad: [0, 0, 0, 0],
            source: None,
            frame_id: None,
            coordinate_space: CoordinateSpace::Image,
            roi_offset: None,
            frame_width: None,
            frame_height: None,
        };

        let boxes = postprocess_output(
            &output,
            &[meta],
            &PostprocessConfig {
                threshold_region: 0.5,
                threshold_kernel: 0.5,
                min_width: 1.0,
                min_height: 1.0,
                max_width_ratio: 0.5,
            },
        );

        assert_eq!(boxes[0].len(), 1);
        assert!((boxes[0][0].x1 - 5.0).abs() < 1e-4);
        assert!((boxes[0][0].x2 - 15.0).abs() < 1e-4);
    }
}
