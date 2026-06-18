use serde::Serialize;

use crate::preprocess::PixelBox;

#[derive(Debug, Clone, Default, Serialize)]
pub struct DetectionMetrics {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub mean_iou: f32,
    pub true_positive_count: usize,
    pub false_positive_count: usize,
    pub false_negative_count: usize,
    pub matched_iou_sum: f32,
}

pub fn bbox_iou(a: PixelBox, b: PixelBox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    let inter = PixelBox { x1, y1, x2, y2 }.area();
    if inter <= 0.0 {
        return 0.0;
    }
    inter / (a.area() + b.area() - inter).max(1e-6)
}

pub fn match_detection_metrics(
    predicted: &[PixelBox],
    ground_truth: &[PixelBox],
    iou_threshold: f32,
) -> DetectionMetrics {
    let mut matched_gt = vec![false; ground_truth.len()];
    let mut true_positive = 0;
    let mut iou_sum = 0.0;
    for pred in predicted {
        let mut best_index = None;
        let mut best_iou = 0.0;
        for (index, gt) in ground_truth.iter().enumerate() {
            if matched_gt[index] {
                continue;
            }
            let iou = bbox_iou(*pred, *gt);
            if iou > best_iou {
                best_iou = iou;
                best_index = Some(index);
            }
        }
        if best_iou >= iou_threshold {
            true_positive += 1;
            iou_sum += best_iou;
            if let Some(index) = best_index {
                matched_gt[index] = true;
            }
        }
    }
    let false_positive = predicted.len().saturating_sub(true_positive);
    let false_negative = ground_truth.len().saturating_sub(true_positive);
    let precision = true_positive as f32 / (true_positive + false_positive).max(1) as f32;
    let recall = true_positive as f32 / (true_positive + false_negative).max(1) as f32;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    DetectionMetrics {
        precision,
        recall,
        f1,
        mean_iou: iou_sum / true_positive.max(1) as f32,
        true_positive_count: true_positive,
        false_positive_count: false_positive,
        false_negative_count: false_negative,
        matched_iou_sum: iou_sum,
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DetectionMetricsAccumulator {
    pub true_positive_count: usize,
    pub false_positive_count: usize,
    pub false_negative_count: usize,
    pub matched_iou_sum: f32,
}

impl DetectionMetricsAccumulator {
    pub fn add(&mut self, metrics: &DetectionMetrics) {
        self.true_positive_count += metrics.true_positive_count;
        self.false_positive_count += metrics.false_positive_count;
        self.false_negative_count += metrics.false_negative_count;
        self.matched_iou_sum += metrics.matched_iou_sum;
    }

    pub fn precision(&self) -> f32 {
        self.true_positive_count as f32
            / (self.true_positive_count + self.false_positive_count).max(1) as f32
    }

    pub fn recall(&self) -> f32 {
        self.true_positive_count as f32
            / (self.true_positive_count + self.false_negative_count).max(1) as f32
    }

    pub fn f1(&self) -> f32 {
        let precision = self.precision();
        let recall = self.recall();
        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }

    pub fn mean_iou(&self) -> f32 {
        self.matched_iou_sum / self.true_positive_count.max(1) as f32
    }
}

pub fn percentile(values: &mut [f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let index = ((values.len() - 1) as f32 * percentile).round() as usize;
    values[index.min(values.len() - 1)]
}
