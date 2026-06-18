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
    let mut stats = vec![ComponentStats::new(width, height); seeds.len()];
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
        let resized_box = PixelBox {
            x1: stat.min_x as f32,
            y1: stat.min_y as f32,
            x2: (stat.max_x + 1) as f32,
            y2: (stat.max_y + 1) as f32,
        };
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
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    confidence_sum: f32,
    count: usize,
}

impl ComponentStats {
    fn new(width: usize, height: usize) -> Self {
        Self {
            min_x: width,
            min_y: height,
            max_x: 0,
            max_y: 0,
            confidence_sum: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, x: usize, y: usize, region_prob: f32, kernel_prob: f32) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.confidence_sum += region_prob * 0.8 + kernel_prob * 0.2;
        self.count += 1;
    }

    fn confidence(&self) -> f32 {
        self.confidence_sum / self.count.max(1) as f32
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
