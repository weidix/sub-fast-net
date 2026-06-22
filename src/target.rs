use crate::preprocess::{PixelBox, Point, RectanglePolygon};

#[derive(Debug, Clone)]
pub struct TargetMasks {
    pub gt_instance: Vec<u32>,
    pub gt_text: Vec<f32>,
    pub gt_kernel: Vec<f32>,
    pub training_mask: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct TargetConfig {
    pub pooling_size: usize,
    pub shrink_kernel_scale: f32,
    pub min_kernel_width: u32,
    pub min_kernel_height: u32,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            pooling_size: 9,
            shrink_kernel_scale: 0.1,
            min_kernel_width: 3,
            min_kernel_height: 3,
        }
    }
}

pub fn generate_targets(
    width: usize,
    height: usize,
    boxes: &[PixelBox],
    ignore_regions: &[PixelBox],
) -> TargetMasks {
    let polygons = boxes.iter().map(PixelBox::to_polygon).collect::<Vec<_>>();
    generate_targets_from_polygons(width, height, &polygons, ignore_regions)
}

pub fn generate_targets_from_polygons(
    width: usize,
    height: usize,
    polygons: &[RectanglePolygon],
    ignore_regions: &[PixelBox],
) -> TargetMasks {
    generate_targets_from_polygons_with_config(
        width,
        height,
        polygons,
        ignore_regions,
        TargetConfig::default(),
    )
}

pub fn generate_targets_with_config(
    width: usize,
    height: usize,
    boxes: &[PixelBox],
    ignore_regions: &[PixelBox],
    config: TargetConfig,
) -> TargetMasks {
    let polygons = boxes.iter().map(PixelBox::to_polygon).collect::<Vec<_>>();
    generate_targets_from_polygons_with_config(width, height, &polygons, ignore_regions, config)
}

pub fn generate_targets_from_polygons_with_config(
    width: usize,
    height: usize,
    polygons: &[RectanglePolygon],
    ignore_regions: &[PixelBox],
    config: TargetConfig,
) -> TargetMasks {
    let len = width * height;
    let mut gt_instance = vec![0_u32; len];
    let mut training_mask = vec![1.0_f32; len];
    for region in ignore_regions {
        fill_f32(&mut training_mask, width, height, *region, 0.0);
    }
    for (index, polygon) in polygons.iter().enumerate() {
        fill_polygon_u32(&mut gt_instance, width, height, polygon, (index + 1) as u32);
    }
    let gt_text = gt_instance
        .iter()
        .map(|value| if *value > 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let gt_kernel = generate_fast_kernel(width, height, polygons, &gt_instance, config);
    TargetMasks {
        gt_instance,
        gt_text,
        gt_kernel,
        training_mask,
    }
}

pub fn shrink_box(bbox: PixelBox, scale: f32) -> PixelBox {
    let cx = (bbox.x1 + bbox.x2) * 0.5;
    let cy = (bbox.y1 + bbox.y2) * 0.5;
    let half_w = bbox.width() * scale * 0.5;
    let half_h = bbox.height() * scale * 0.5;
    PixelBox {
        x1: cx - half_w,
        y1: cy - half_h,
        x2: cx + half_w,
        y2: cy + half_h,
    }
}

fn generate_fast_kernel(
    width: usize,
    height: usize,
    polygons: &[RectanglePolygon],
    gt_instance: &[u32],
    config: TargetConfig,
) -> Vec<f32> {
    let len = width * height;
    let mut kernel = vec![0.0_f32; len];
    let pooled = min_pool_instances(width, height, gt_instance, config.pooling_size);
    for (index, value) in pooled.iter().enumerate() {
        if *value > 0 {
            kernel[index] = 1.0;
        }
    }

    let mut owner = vec![0_u32; len];
    for (index, polygon) in polygons.iter().enumerate() {
        let instance_id = (index + 1) as u32;
        let shrink = shrink_polygon_preserving_min_size(
            polygon,
            config.shrink_kernel_scale,
            config.min_kernel_width,
            config.min_kernel_height,
        );
        let (x1, y1, x2, y2) = raster_bounds(width, height, shrink.bounding_box());
        for y in y1..y2 {
            for x in x1..x2 {
                if !point_in_polygon(
                    Point {
                        x: x as f32 + 0.5,
                        y: y as f32 + 0.5,
                    },
                    &shrink.points,
                ) {
                    continue;
                }
                let offset = y * width + x;
                if owner[offset] == 0 || owner[offset] == instance_id {
                    owner[offset] = instance_id;
                    kernel[offset] = 1.0;
                } else {
                    kernel[offset] = 0.0;
                }
            }
        }
    }
    kernel
}

fn min_pool_instances(
    width: usize,
    height: usize,
    instances: &[u32],
    pooling_size: usize,
) -> Vec<u32> {
    let radius = pooling_size / 2;
    let mut output = vec![0_u32; width * height];
    for y in 0..height {
        for x in 0..width {
            let instance_id = instances[y * width + x];
            if instance_id == 0 {
                continue;
            }
            let x1 = x.saturating_sub(radius);
            let y1 = y.saturating_sub(radius);
            let x2 = (x + radius + 1).min(width);
            let y2 = (y + radius + 1).min(height);
            let mut keep = true;
            'outer: for yy in y1..y2 {
                for xx in x1..x2 {
                    if instances[yy * width + xx] != instance_id {
                        keep = false;
                        break 'outer;
                    }
                }
            }
            if keep {
                output[y * width + x] = instance_id;
            }
        }
    }
    output
}

fn shrink_box_preserving_min_size(
    bbox: PixelBox,
    scale: f32,
    min_width: u32,
    min_height: u32,
) -> PixelBox {
    let target_width = shrink_dimension_preserving_minimum(bbox.width(), scale, min_width);
    let target_height = shrink_dimension_preserving_minimum(bbox.height(), scale, min_height);
    let cx = (bbox.x1 + bbox.x2) * 0.5;
    let cy = (bbox.y1 + bbox.y2) * 0.5;
    PixelBox {
        x1: cx - target_width * 0.5,
        y1: cy - target_height * 0.5,
        x2: cx + target_width * 0.5,
        y2: cy + target_height * 0.5,
    }
}

fn shrink_dimension_preserving_minimum(size: f32, scale: f32, minimum: u32) -> f32 {
    if size <= minimum as f32 {
        size
    } else {
        (size * scale).max(minimum as f32).min(size)
    }
}

fn shrink_polygon_preserving_min_size(
    polygon: &RectanglePolygon,
    scale: f32,
    min_width: u32,
    min_height: u32,
) -> RectanglePolygon {
    let bbox = polygon.bounding_box();
    if bbox.width() <= min_width as f32 && bbox.height() <= min_height as f32 {
        return polygon.clone();
    }
    let bbox_shrink = shrink_box_preserving_min_size(bbox, scale, min_width, min_height);
    let sx = (bbox_shrink.width() / bbox.width()).clamp(0.0, 1.0);
    let sy = (bbox_shrink.height() / bbox.height()).clamp(0.0, 1.0);
    let cx = (bbox.x1 + bbox.x2) * 0.5;
    let cy = (bbox.y1 + bbox.y2) * 0.5;
    RectanglePolygon {
        points: polygon
            .points
            .iter()
            .map(|point| Point {
                x: cx + (point.x - cx) * sx,
                y: cy + (point.y - cy) * sy,
            })
            .collect(),
    }
}

fn fill_f32(mask: &mut [f32], width: usize, height: usize, bbox: PixelBox, value: f32) {
    let (x1, y1, x2, y2) = raster_bounds(width, height, bbox);
    for y in y1..y2 {
        for x in x1..x2 {
            mask[y * width + x] = value;
        }
    }
}

fn fill_polygon_u32(
    mask: &mut [u32],
    width: usize,
    height: usize,
    polygon: &RectanglePolygon,
    value: u32,
) {
    let (x1, y1, x2, y2) = raster_bounds(width, height, polygon.bounding_box());
    for y in y1..y2 {
        for x in x1..x2 {
            if point_in_polygon(
                Point {
                    x: x as f32 + 0.5,
                    y: y as f32 + 0.5,
                },
                &polygon.points,
            ) {
                mask[y * width + x] = value;
            }
        }
    }
}

fn raster_bounds(width: usize, height: usize, bbox: PixelBox) -> (usize, usize, usize, usize) {
    let x1 = bbox.x1.floor().clamp(0.0, width as f32) as usize;
    let y1 = bbox.y1.floor().clamp(0.0, height as f32) as usize;
    let x2 = bbox.x2.ceil().clamp(0.0, width as f32) as usize;
    let y2 = bbox.y2.ceil().clamp(0.0, height as f32) as usize;
    (x1, y1, x2, y2)
}

fn point_in_polygon(point: Point, polygon: &[Point]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut prev = polygon[polygon.len() - 1];
    for current in polygon {
        let crosses_y = (current.y > point.y) != (prev.y > point.y);
        if crosses_y {
            let x_at_y =
                (prev.x - current.x) * (point.y - current.y) / (prev.y - current.y) + current.x;
            if point.x < x_at_y {
                inside = !inside;
            }
        }
        prev = *current;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::{
        Point, RectanglePolygon, TargetConfig, generate_targets_from_polygons_with_config,
    };
    use crate::preprocess::PixelBox;

    #[test]
    fn kernel_shrinks_wide_subtitle_even_when_height_is_below_minimum() {
        let polygon = RectanglePolygon {
            points: vec![
                Point { x: 10.0, y: 10.0 },
                Point { x: 90.0, y: 10.0 },
                Point { x: 90.0, y: 30.0 },
                Point { x: 10.0, y: 30.0 },
            ],
        };

        let targets = generate_targets_from_polygons_with_config(
            100,
            40,
            &[polygon],
            &[],
            TargetConfig {
                pooling_size: 9,
                shrink_kernel_scale: 0.1,
                min_kernel_width: 3,
                min_kernel_height: 55,
            },
        );

        let text_pixels = targets.gt_text.iter().filter(|value| **value > 0.0).count();
        let kernel_pixels = targets
            .gt_kernel
            .iter()
            .filter(|value| **value > 0.0)
            .count();

        assert!(text_pixels > 0);
        assert!(
            kernel_pixels < text_pixels,
            "kernel should remain horizontally shrunk for subtitle-like boxes"
        );
    }

    #[test]
    fn tiny_boxes_do_not_expand_to_minimum_kernel_size() {
        let bbox = PixelBox {
            x1: 10.0,
            y1: 10.0,
            x2: 12.0,
            y2: 12.0,
        };

        let targets = generate_targets_from_polygons_with_config(
            32,
            32,
            &[bbox.to_polygon()],
            &[],
            TargetConfig {
                pooling_size: 9,
                shrink_kernel_scale: 0.1,
                min_kernel_width: 3,
                min_kernel_height: 55,
            },
        );

        let text_pixels = targets.gt_text.iter().filter(|value| **value > 0.0).count();
        let kernel_pixels = targets
            .gt_kernel
            .iter()
            .filter(|value| **value > 0.0)
            .count();

        assert_eq!(kernel_pixels, text_pixels);
    }
}
