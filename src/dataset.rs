use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    config::TrainConfig,
    preprocess::{PixelBox, RectanglePolygon, YoloBox, yolo_to_pixel},
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnnotationMeta {
    pub image: Option<String>,
    pub source_video: Option<String>,
    pub frame_index: Option<u64>,
    pub image_width: Option<u32>,
    pub image_height: Option<u32>,
    pub roi_offset: Option<[i32; 2]>,
    pub filter_region: Option<[f32; 4]>,
    pub raw: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RawLabelMaskFile {
    pub version: Option<u32>,
    pub description: Option<String>,
    pub items: HashMap<String, HashMap<String, RawLabelMaskRecord>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RawLabelMaskRecord {
    pub drop_image: Option<bool>,
    pub masked: Option<bool>,
    pub deleted: Option<bool>,
    pub unreliable: Option<bool>,
    pub exclude_from_loss: Option<bool>,
    pub reason: Option<String>,
    pub updated_at: Option<u64>,
    #[serde(default)]
    pub bbox: Option<[f32; 4]>,
    #[serde(default)]
    pub add_bbox: Option<[f32; 4]>,
    #[serde(default)]
    pub ignore_region: Option<[f32; 4]>,
    #[serde(default)]
    pub action: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ApplyLabelMaskResult {
    pub dropped: bool,
    pub deleted_count: usize,
    pub corrected_count: usize,
    pub added_count: usize,
    pub unreliable_count: usize,
    pub ignore_regions: Vec<PixelBox>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Val,
}

#[derive(Debug, Clone)]
pub struct DatasetRoot {
    pub root_id: usize,
    pub path: PathBuf,
    pub annotations: HashMap<String, AnnotationMeta>,
    pub label_masks: RawLabelMaskFile,
}

#[derive(Debug, Clone)]
pub struct SampleIndex {
    pub root_id: usize,
    pub sample_id: String,
    pub image_path: PathBuf,
    pub label_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct DatasetSample {
    pub image_path: PathBuf,
    pub label_path: PathBuf,
    pub root_id: usize,
    pub sample_id: String,
    pub original_width: u32,
    pub original_height: u32,
    pub yolo_boxes_raw: Vec<YoloBox>,
    pub pixel_boxes_raw: Vec<PixelBox>,
    pub pixel_boxes_after_label_masks: Vec<PixelBox>,
    pub rectangle_polygons: Vec<RectanglePolygon>,
    pub ignore_regions: Vec<PixelBox>,
    pub annotation_meta: Option<AnnotationMeta>,
    pub source: Option<String>,
    pub frame_id: Option<String>,
    pub issues: Vec<String>,
    pub ignored: bool,
    pub ignore_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SubtitleDataset {
    pub split: DatasetSplit,
    pub strict: bool,
    pub roots: Vec<DatasetRoot>,
    samples: Vec<SampleIndex>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct DatasetInspectionReport {
    pub total_sample_count: usize,
    pub sample_count_per_root: Vec<usize>,
    pub empty_label_count: usize,
    pub invalid_label_count: usize,
    pub bboxes_deleted_by_label_masks: usize,
    pub bboxes_corrected_by_label_masks: usize,
    pub bboxes_added_by_label_masks: usize,
    pub unreliable_label_count: usize,
    pub dropped_image_count: usize,
    pub ignore_region_count: usize,
    pub image_size_distribution: HashMap<String, usize>,
    pub bbox_size_distribution: HashMap<String, usize>,
    pub subtitle_position_distribution: HashMap<String, usize>,
    pub abnormal_samples: Vec<String>,
}

impl SubtitleDataset {
    pub fn from_train_config(config: &TrainConfig) -> Result<Self> {
        let roots = config
            .train_roots
            .iter()
            .enumerate()
            .map(|(root_id, path)| load_root(root_id, Path::new(path), config.strict_dataset))
            .collect::<Result<Vec<_>>>()?;
        let mut dataset = Self::from_roots(DatasetSplit::Train, config.strict_dataset, roots)?;
        if let Some(max) = config.max_train_samples {
            dataset.samples = match config.train_empty_sample_ratio {
                Some(ratio) => balanced_sample_limit_with_empty_ratio(&dataset.samples, max, ratio),
                None => balanced_sample_limit(&dataset.samples, max),
            };
        }
        Ok(dataset)
    }

    pub fn from_val_config(config: &TrainConfig) -> Result<Self> {
        let roots = vec![load_root(
            0,
            Path::new(&config.val_root),
            config.strict_dataset,
        )?];
        let mut dataset = Self::from_roots(DatasetSplit::Val, config.strict_dataset, roots)?;
        if let Some(max) = config.max_val_samples {
            dataset.samples = balanced_sample_limit(&dataset.samples, max);
        }
        Ok(dataset)
    }

    pub fn from_roots(split: DatasetSplit, strict: bool, roots: Vec<DatasetRoot>) -> Result<Self> {
        let mut samples = Vec::new();
        for root in &roots {
            let images_dir = root.path.join("images");
            let labels_dir = root.path.join("labels");
            if !images_dir.is_dir() {
                if strict {
                    bail!("missing images directory {}", images_dir.display());
                }
                continue;
            }
            for entry in fs::read_dir(&images_dir)
                .with_context(|| format!("failed to read {}", images_dir.display()))?
            {
                let entry = entry?;
                let path = entry.path();
                let Some(ext) = path.extension().and_then(|value| value.to_str()) else {
                    continue;
                };
                if !ext.eq_ignore_ascii_case("jpg") && !ext.eq_ignore_ascii_case("jpeg") {
                    continue;
                }
                let stem = path
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| anyhow!("invalid image file name {}", path.display()))?
                    .to_string();
                samples.push(SampleIndex {
                    root_id: root.root_id,
                    sample_id: stem.clone(),
                    image_path: path,
                    label_path: labels_dir.join(format!("{stem}.txt")),
                });
            }
        }
        samples.sort_by(|a, b| {
            a.root_id
                .cmp(&b.root_id)
                .then_with(|| a.sample_id.cmp(&b.sample_id))
        });
        Ok(Self {
            split,
            strict,
            roots,
            samples,
        })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn sample_indices(&self) -> &[SampleIndex] {
        &self.samples
    }

    pub fn load_sample(&self, index: usize) -> Result<DatasetSample> {
        let sample = self
            .samples
            .get(index)
            .ok_or_else(|| anyhow!("sample index out of bounds: {index}"))?;
        let root = self
            .roots
            .iter()
            .find(|root| root.root_id == sample.root_id)
            .ok_or_else(|| anyhow!("missing root {}", sample.root_id))?;
        let mut issues = Vec::new();
        if !sample.label_path.is_file() {
            if self.strict {
                bail!("missing label file {}", sample.label_path.display());
            }
            issues.push(format!("missing label {}", sample.label_path.display()));
        }

        let (width, height) = read_image_size(&sample.image_path)?;
        let annotation_meta = root.annotations.get(&sample.sample_id).cloned();
        if let Some(meta) = &annotation_meta
            && let (Some(w), Some(h)) = (meta.image_width, meta.image_height)
            && (w, h) != (width, height)
        {
            let msg = format!(
                "annotation image size mismatch for {}: annotation={}x{}, image={}x{}",
                sample.sample_id, w, h, width, height
            );
            if self.strict {
                bail!(msg);
            }
            issues.push(msg);
        }

        let yolo_boxes_raw = if sample.label_path.is_file() {
            parse_yolo_label_file(&sample.label_path, self.strict)
                .with_context(|| format!("failed parsing {}", sample.label_path.display()))?
        } else {
            Vec::new()
        };
        let pixel_boxes_raw = yolo_boxes_raw
            .iter()
            .map(|bbox| yolo_to_pixel(*bbox, width, height))
            .collect::<Result<Vec<_>>>()?;
        let mask_result = apply_label_masks(
            &root.label_masks,
            &sample.sample_id,
            &pixel_boxes_raw,
            width,
            height,
            self.strict,
        )?;
        let rectangle_polygons = mask_result
            .0
            .iter()
            .map(|bbox| bbox.to_polygon())
            .collect::<Vec<_>>();
        let source = annotation_meta
            .as_ref()
            .and_then(|meta| meta.source_video.clone());
        let frame_id = annotation_meta
            .as_ref()
            .and_then(|meta| meta.frame_index.map(|value| value.to_string()));
        let ignored = mask_result.1.dropped;
        let mut ignore_reason = None;
        if mask_result.1.dropped {
            let msg = "drop_image label mask".to_string();
            if self.strict {
                bail!("{msg} for {}", sample.sample_id);
            }
            issues.push(msg.clone());
            ignore_reason = Some(msg);
        }
        Ok(DatasetSample {
            image_path: sample.image_path.clone(),
            label_path: sample.label_path.clone(),
            root_id: sample.root_id,
            sample_id: sample.sample_id.clone(),
            original_width: width,
            original_height: height,
            yolo_boxes_raw,
            pixel_boxes_raw,
            pixel_boxes_after_label_masks: mask_result.0,
            rectangle_polygons,
            ignore_regions: mask_result.1.ignore_regions,
            annotation_meta,
            source,
            frame_id,
            issues,
            ignored,
            ignore_reason,
        })
    }

    pub fn inspect(&self) -> DatasetInspectionReport {
        let mut report = DatasetInspectionReport {
            total_sample_count: self.len(),
            sample_count_per_root: vec![0; self.roots.len()],
            ..DatasetInspectionReport::default()
        };
        for sample in &self.samples {
            if let Some(count) = report.sample_count_per_root.get_mut(sample.root_id) {
                *count += 1;
            }
        }
        for index in 0..self.len() {
            match self.load_sample(index) {
                Ok(sample) => {
                    if sample.yolo_boxes_raw.is_empty() {
                        report.empty_label_count += 1;
                    }
                    let size_key = format!("{}x{}", sample.original_width, sample.original_height);
                    *report.image_size_distribution.entry(size_key).or_insert(0) += 1;
                    let root = self
                        .roots
                        .iter()
                        .find(|root| root.root_id == sample.root_id)
                        .expect("sample root should exist");
                    if let Ok((_boxes, masks)) = apply_label_masks(
                        &root.label_masks,
                        &sample.sample_id,
                        &sample.pixel_boxes_raw,
                        sample.original_width,
                        sample.original_height,
                        self.strict,
                    ) {
                        report.bboxes_deleted_by_label_masks += masks.deleted_count;
                        report.bboxes_corrected_by_label_masks += masks.corrected_count;
                        report.bboxes_added_by_label_masks += masks.added_count;
                        report.unreliable_label_count += masks.unreliable_count;
                        report.dropped_image_count += usize::from(masks.dropped);
                    }
                    report.ignore_region_count += sample.ignore_regions.len();
                    for bbox in &sample.pixel_boxes_after_label_masks {
                        *report
                            .bbox_size_distribution
                            .entry(bbox.size_bucket())
                            .or_insert(0) += 1;
                        *report
                            .subtitle_position_distribution
                            .entry(bbox.vertical_bucket(sample.original_height))
                            .or_insert(0) += 1;
                    }
                    report.abnormal_samples.extend(sample.issues);
                }
                Err(err) => {
                    report.invalid_label_count += 1;
                    report.abnormal_samples.push(err.to_string());
                }
            }
        }
        report
    }
}

fn balanced_sample_limit(samples: &[SampleIndex], max: usize) -> Vec<SampleIndex> {
    if max >= samples.len() {
        return samples.to_vec();
    }
    let mut by_root = BTreeMap::<usize, (Vec<SampleIndex>, Vec<SampleIndex>)>::new();
    for sample in samples {
        let (labeled, unlabeled) = by_root.entry(sample.root_id).or_default();
        if sample_has_nonempty_label(sample) {
            labeled.push(sample.clone());
        } else {
            unlabeled.push(sample.clone());
        }
    }
    for (labeled, unlabeled) in by_root.values_mut() {
        spread_sample_order(labeled);
        spread_sample_order(unlabeled);
    }
    let mut labeled_cursors = BTreeMap::<usize, usize>::new();
    let mut unlabeled_cursors = BTreeMap::<usize, usize>::new();
    let mut limited = Vec::with_capacity(max);
    for prefer_labeled in [true, false] {
        while limited.len() < max {
            let mut pushed = false;
            for (root_id, (labeled_samples, unlabeled_samples)) in &by_root {
                if limited.len() >= max {
                    break;
                }
                let (root_samples, cursors) = if prefer_labeled {
                    (labeled_samples, &mut labeled_cursors)
                } else {
                    (unlabeled_samples, &mut unlabeled_cursors)
                };
                let cursor = cursors.entry(*root_id).or_default();
                if let Some(sample) = root_samples.get(*cursor) {
                    limited.push(sample.clone());
                    *cursor += 1;
                    pushed = true;
                }
            }
            if !pushed {
                break;
            }
        }
    }
    limited
}

fn balanced_sample_limit_with_empty_ratio(
    samples: &[SampleIndex],
    max: usize,
    empty_ratio: f32,
) -> Vec<SampleIndex> {
    if max >= samples.len() {
        return samples.to_vec();
    }
    let empty_target = ((max as f32) * empty_ratio).round() as usize;
    let labeled_target = max.saturating_sub(empty_target);
    let mut by_root = BTreeMap::<usize, (Vec<SampleIndex>, Vec<SampleIndex>)>::new();
    for sample in samples {
        let (labeled, empty) = by_root.entry(sample.root_id).or_default();
        if sample_has_nonempty_label(sample) {
            labeled.push(sample.clone());
        } else {
            empty.push(sample.clone());
        }
    }
    for (labeled, empty) in by_root.values_mut() {
        spread_sample_order(labeled);
        spread_sample_order(empty);
    }

    let mut labeled = Vec::with_capacity(labeled_target);
    let mut empty = Vec::with_capacity(empty_target);
    push_balanced_kind(&by_root, &mut labeled, labeled_target, true);
    push_balanced_kind(&by_root, &mut empty, empty_target, false);
    if labeled.len() + empty.len() < max {
        let remaining = max - labeled.len() - empty.len();
        push_balanced_kind(&by_root, &mut labeled, remaining, true);
    }
    if labeled.len() + empty.len() < max {
        let remaining = max - labeled.len() - empty.len();
        push_balanced_kind(&by_root, &mut empty, remaining, false);
    }
    let mut limited = interleave_labeled_and_empty(labeled, empty);
    limited.truncate(max);
    limited
}

fn interleave_labeled_and_empty(
    labeled: Vec<SampleIndex>,
    empty: Vec<SampleIndex>,
) -> Vec<SampleIndex> {
    if labeled.is_empty() {
        return empty;
    }
    if empty.is_empty() {
        return labeled;
    }
    let mut limited = Vec::with_capacity(labeled.len() + empty.len());
    let mut empty_iter = empty.into_iter();
    let empty_count = empty_iter.len();
    let gap = (labeled.len() / empty_count.max(1)).max(1);
    for (index, sample) in labeled.into_iter().enumerate() {
        limited.push(sample);
        if (index + 1).is_multiple_of(gap)
            && let Some(empty_sample) = empty_iter.next()
        {
            limited.push(empty_sample);
        }
    }
    limited.extend(empty_iter);
    limited
}

fn push_balanced_kind(
    by_root: &BTreeMap<usize, (Vec<SampleIndex>, Vec<SampleIndex>)>,
    limited: &mut Vec<SampleIndex>,
    count: usize,
    labeled: bool,
) {
    let initial_len = limited.len();
    let target_len = initial_len + count;
    let mut cursors = BTreeMap::<usize, usize>::new();
    while limited.len() < target_len {
        let mut pushed = false;
        for (root_id, (labeled_samples, empty_samples)) in by_root {
            if limited.len() >= target_len {
                break;
            }
            let root_samples = if labeled {
                labeled_samples
            } else {
                empty_samples
            };
            let cursor = cursors.entry(*root_id).or_default();
            while let Some(sample) = root_samples.get(*cursor) {
                *cursor += 1;
                if limited.iter().any(|existing| {
                    existing.root_id == sample.root_id && existing.sample_id == sample.sample_id
                }) {
                    continue;
                }
                limited.push(sample.clone());
                pushed = true;
                break;
            }
        }
        if !pushed {
            break;
        }
    }
}

fn spread_sample_order(samples: &mut [SampleIndex]) {
    samples.sort_by_key(|sample| deterministic_spread_key(sample.root_id, &sample.sample_id));
}

fn deterministic_spread_key(root_id: usize, sample_id: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64 ^ root_id as u64;
    for byte in sample_id.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn sample_has_nonempty_label(sample: &SampleIndex) -> bool {
    fs::metadata(&sample.label_path)
        .map(|metadata| metadata.len() > 0)
        .unwrap_or(false)
}

impl burn_dataset::Dataset<DatasetSample> for SubtitleDataset {
    fn get(&self, index: usize) -> Option<DatasetSample> {
        self.load_sample(index).ok()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

pub fn load_root(root_id: usize, path: &Path, strict: bool) -> Result<DatasetRoot> {
    if !path.is_dir() {
        bail!("dataset root does not exist: {}", path.display());
    }
    let annotations = load_annotations(&path.join("annotations.jsonl"), strict)?;
    let label_masks = load_label_masks(&path.join("label_masks.json"), strict)?;
    Ok(DatasetRoot {
        root_id,
        path: path.to_path_buf(),
        annotations,
        label_masks,
    })
}

pub fn parse_yolo_label_file(path: &Path, strict: bool) -> Result<Vec<YoloBox>> {
    let text = fs::read_to_string(path)?;
    parse_yolo_label_text(&text, strict)
}

pub fn parse_yolo_label_text(text: &str, strict: bool) -> Result<Vec<YoloBox>> {
    let mut boxes = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts = trimmed.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 5 {
            if strict {
                bail!("line {} must contain 5 columns", line_index + 1);
            }
            continue;
        }
        let class_id = parts[0]
            .parse::<i64>()
            .with_context(|| format!("invalid class id on line {}", line_index + 1))?;
        let parsed = parts[1].parse::<f32>().and_then(|x_center| {
            Ok((
                x_center,
                parts[2].parse::<f32>()?,
                parts[3].parse::<f32>()?,
                parts[4].parse::<f32>()?,
            ))
        });
        let (x_center, y_center, width, height) = match parsed {
            Ok(values) => values,
            Err(err) => {
                if strict {
                    return Err(err).with_context(|| {
                        format!("invalid bbox floats on line {}", line_index + 1)
                    });
                }
                continue;
            }
        };
        let bbox = YoloBox {
            class_id,
            x_center,
            y_center,
            width,
            height,
        };
        if let Err(err) = bbox.validate() {
            if strict {
                return Err(err)
                    .with_context(|| format!("invalid bbox on line {}", line_index + 1));
            }
            continue;
        }
        boxes.push(bbox);
    }
    Ok(boxes)
}

pub fn load_annotations(path: &Path, strict: bool) -> Result<HashMap<String, AnnotationMeta>> {
    if !path.is_file() {
        if strict {
            bail!("missing annotations.jsonl {}", path.display());
        }
        return Ok(HashMap::new());
    }
    let text = fs::read_to_string(path)?;
    let mut result = HashMap::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(err) => {
                if strict {
                    return Err(err).with_context(|| {
                        format!("invalid annotation json on line {}", line_index + 1)
                    });
                }
                continue;
            }
        };
        let image = value
            .get("image")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let key = image
            .as_deref()
            .and_then(|value| Path::new(value).file_stem())
            .and_then(|value| value.to_str())
            .map(ToOwned::to_owned);
        let Some(key) = key else {
            if strict {
                bail!("annotation line {} lacks image stem", line_index + 1);
            }
            continue;
        };
        result.insert(
            key,
            AnnotationMeta {
                image,
                source_video: value
                    .get("source_video")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                frame_index: value.get("frame_index").and_then(Value::as_u64),
                image_width: value
                    .get("image_width")
                    .and_then(Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok()),
                image_height: value
                    .get("image_height")
                    .and_then(Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok()),
                roi_offset: parse_i32_pair(value.get("roi_offset")),
                filter_region: parse_f32_quad(value.get("filter_region")),
                raw: value,
            },
        );
    }
    Ok(result)
}

pub fn load_label_masks(path: &Path, strict: bool) -> Result<RawLabelMaskFile> {
    if !path.is_file() {
        if strict {
            bail!("missing label_masks.json {}", path.display());
        }
        return Ok(RawLabelMaskFile::default());
    }
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).with_context(|| format!("failed parsing {}", path.display()))
}

pub fn apply_label_masks(
    masks: &RawLabelMaskFile,
    sample_id: &str,
    raw_boxes: &[PixelBox],
    width: u32,
    height: u32,
    strict: bool,
) -> Result<(Vec<PixelBox>, ApplyLabelMaskResult)> {
    let mut result = ApplyLabelMaskResult::default();
    let mut boxes = raw_boxes.to_vec();
    let Some(items) = masks.items.get(sample_id) else {
        return Ok((boxes, result));
    };
    let mut deleted = Vec::new();
    for (index_text, record) in items {
        let action = record.action.as_deref().unwrap_or_default();
        if record.drop_image.unwrap_or(false) || action.eq_ignore_ascii_case("drop_image") {
            result.dropped = true;
            boxes.clear();
            result.ignore_regions.push(PixelBox {
                x1: 0.0,
                y1: 0.0,
                x2: width as f32,
                y2: height as f32,
            });
            continue;
        }
        let index = match index_text.parse::<usize>() {
            Ok(index) => index,
            Err(err) => {
                if index_text.eq_ignore_ascii_case("add")
                    || index_text.eq_ignore_ascii_case("ignore")
                    || index_text.starts_with("add_")
                    || index_text.starts_with("ignore_")
                {
                    usize::MAX
                } else if strict {
                    return Err(err).with_context(|| {
                        format!("invalid label mask index {index_text} for {sample_id}")
                    });
                } else {
                    usize::MAX
                }
            }
        };
        let delete = record.masked.unwrap_or(false)
            || record.deleted.unwrap_or(false)
            || action.eq_ignore_ascii_case("delete_bbox")
            || action.eq_ignore_ascii_case("delete");
        if delete {
            if index < boxes.len() {
                deleted.push(index);
            } else if strict && index != usize::MAX {
                bail!("label mask index {index} out of bounds for {sample_id}");
            }
        }
        if let Some(ignore_region) = record.ignore_region {
            let bbox = PixelBox::from_array(ignore_region).clip(width, height);
            if bbox.is_valid() {
                result.ignore_regions.push(bbox);
            }
        }
        if record.unreliable.unwrap_or(false) || record.exclude_from_loss.unwrap_or(false) {
            if index < boxes.len() {
                result.ignore_regions.push(boxes[index]);
                result.unreliable_count += 1;
            } else if strict && index != usize::MAX {
                bail!("label mask unreliable index {index} out of bounds for {sample_id}");
            }
        }
        if let Some(corrected) = record.bbox {
            if action.eq_ignore_ascii_case("add_bbox") || index == usize::MAX {
                let bbox = PixelBox::from_array(corrected).clip(width, height);
                if bbox.is_valid() {
                    boxes.push(bbox);
                    result.added_count += 1;
                }
            } else if index < boxes.len() {
                boxes[index] = PixelBox::from_array(corrected).clip(width, height);
                result.corrected_count += 1;
            } else if strict {
                bail!("label mask correction index {index} out of bounds for {sample_id}");
            }
        }
        if let Some(added) = record.add_bbox {
            let bbox = PixelBox::from_array(added).clip(width, height);
            if bbox.is_valid() {
                boxes.push(bbox);
                result.added_count += 1;
            }
        }
    }
    deleted.sort_unstable();
    deleted.dedup();
    for index in deleted.into_iter().rev() {
        boxes.remove(index);
        result.deleted_count += 1;
    }
    Ok((boxes, result))
}

fn read_image_size(path: &Path) -> Result<(u32, u32)> {
    let reader = image::ImageReader::open(path)
        .with_context(|| format!("failed to open image {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("failed to guess image format {}", path.display()))?;
    reader
        .into_dimensions()
        .with_context(|| format!("failed to read image dimensions {}", path.display()))
}

fn parse_i32_pair(value: Option<&Value>) -> Option<[i32; 2]> {
    let values = value?.as_array()?;
    Some([
        values.first()?.as_i64()? as i32,
        values.get(1)?.as_i64()? as i32,
    ])
}

fn parse_f32_quad(value: Option<&Value>) -> Option<[f32; 4]> {
    let values = value?.as_array()?;
    Some([
        values.first()?.as_f64()? as f32,
        values.get(1)?.as_f64()? as f32,
        values.get(2)?.as_f64()? as f32,
        values.get(3)?.as_f64()? as f32,
    ])
}
