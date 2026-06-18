# SubFastNet Project Generation Requirements

## 1. Project Positioning

You are a senior Rust / Burn deep learning engineering agent. Generate a complete, engineering-grade, trainable, verifiable, inferable, and benchmarkable Rust Burn project for training a high-FPS model focused on subtitle region detection.

Project name: SubFastNet

Task type: subtitle region detection

This is not an OCR project. Do not perform character recognition, subtitle text recognition, end-to-end OCR, CTC, attention decoding, or text decoding.

The input is an image.

The output is the subtitle region bbox in pixel coordinates plus confidence.

Default inference output format:

```json
{
  "image": "path/to/image.jpg",
  "width": 1920,
  "height": 1080,
  "boxes": [
    {
      "x1": 100,
      "y1": 720,
      "x2": 820,
      "y2": 770,
      "confidence": 0.94
    }
  ],
  "meta": {
    "source": "optional",
    "frame_id": "optional"
  }
}
```

The core project goal is to train a model that follows the ideas of FAST / TextNet, but is simplified and optimized specifically for subtitle detection scenarios. The priority is high-FPS desktop inference while supporting CPU, CUDA, and WGPU training and inference.

## 2. Confirmed Facts

The following points are already clear. Do not ask the user to confirm them again:

1. The task is subtitle region detection, not OCR.
2. Character recognition is not required.
3. OCR metrics such as CER, WER, or text exact match are not required.
4. All bboxes are subtitle boxes.
5. This is a single-class detection model.
6. `labels/*.txt` uses YOLO bbox format.
7. Two-line subtitles are annotated as one bbox per line.
8. The model learns from the bboxes in the training samples and does not need to merge lines.
9. The training set consists of multiple roots.
10. The validation set has only one root.
11. Each root has the same internal structure.
12. `label_masks.json` must participate in training-sample injection.
13. `annotations.jsonl` must be read and used as metadata for data validation, error analysis, and output.
14. Inference output uses bbox pixel coordinates.
15. The deployment target is mainly desktop. The high-priority goal is faster inference FPS.
16. The model name must be `SubFastNet`.
17. The default model architecture must reference the `TextNet` used by FAST.
18. Do not default to MobileNet, LCNet, ShuffleNet, a YOLO backbone, or any other generic classification backbone.

## 3. Dataset Structure

The training set supports multiple roots.

The validation set supports one root.

Each root has the following fixed structure:

```text
dataset_root/
|-- images/
|   `-- *.jpg
|-- labels/
|   `-- *.txt
|-- annotations.jsonl
`-- label_masks.json
```

`images/` stores jpg images.

`labels/` stores txt label files with the same base name as the images.

`annotations.jsonl` stores metadata for each image, such as source, size, frame id, and detection boxes.

`label_masks.json` stores manually masked, deleted, corrected, or supplemented label records.

The training config must support multiple training roots, for example:

```toml
train_roots = [
  "/data/subtitle/train_a",
  "/data/subtitle/train_b",
  "/data/subtitle/train_c"
]

val_root = "/data/subtitle/val"
```

## 4. Label Format

`labels/*.txt` uses YOLO bbox format.

Each line represents one subtitle box:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.404167 0.962963 0.158333 0.066667
0 0.624479 0.961574 0.098958 0.065741
```

Field meanings:

* `class_id`: class ID, always treated as the subtitle class.
* `x_center`: bbox center x in normalized coordinates.
* `y_center`: bbox center y in normalized coordinates.
* `width`: bbox width in normalized coordinates.
* `height`: bbox height in normalized coordinates.

All bboxes are subtitle regions.

This is a single-class model and does not need a multi-class classification head.

Parsing rules:

1. Each line must have 5 columns.
2. `class_id` must parse as an integer.
3. `x_center`, `y_center`, `width`, and `height` must parse as floating-point values.
4. `width > 0`.
5. `height > 0`.
6. The bbox should be within a reasonable range.
7. Out-of-bounds bboxes may error or be clipped depending on the strict policy.
8. An empty label file means the image has no subtitle samples, unless the config requires strict errors.
9. Missing labels are handled according to the strict policy.

## 5. `label_masks.json` Handling Requirements

`label_masks.json` must participate in training-sample injection and must not be ignored.

When loading a sample, processing must happen in this order:

1. Read the image.
2. Read the same-name `labels/*.txt`.
3. Read the corresponding image metadata from `annotations.jsonl`.
4. Read the manual masking or correction rules for the image from `label_masks.json`.
5. Apply `label_masks.json` first.
6. Then generate the final training target.

`label_masks.json` may contain the following actions:

* Ignore an entire image.
* Delete a bbox.
* Correct a bbox.
* Add a manually supplemented bbox.
* Add an ignore region.
* Mark certain regions as excluded from loss.
* Mark certain original labels as unreliable.

The agent must read real files and implement adaptation according to the schema. Only ask the user for confirmation when the `label_masks.json` schema cannot be inferred from file contents, contains conflicts, or would change the training-label semantics.

Recommended compatibility layer:

```text
RawLabelMaskRecord
LabelMaskAction
ApplyLabelMaskResult
```

Recommended processing priority:

1. drop image
2. remove bbox
3. correct bbox
4. add bbox
5. add ignore region
6. final validation

`label_masks.json` is applied after raw labels and before target-mask generation.

The bboxes that enter training must be the bboxes after `label_masks.json` processing.

Ignore regions must enter `training_mask` and be excluded from loss computation.

## 6. `annotations.jsonl` Handling Requirements

`annotations.jsonl` must be read.

By default, training labels are based on `labels/*.txt`.

`annotations.jsonl` does not directly override labels unless the real data contains explicit override rules.

Uses of `annotations.jsonl`:

1. Validate image size.
2. Record source.
3. Record frame_id.
4. Record video source.
5. Record original detection-box metadata.
6. Support error-sample analysis.
7. Supplement inference-output metadata.
8. Support dataset inspection reports.
9. Support debugging and statistical analysis.

If the `annotations.jsonl` schema can be inferred from file contents, the agent should implement the adapter directly.

If the schema cannot be inferred, or if its fields would change label semantics, ask the user for confirmation.

## 7. Dataset Design

The implementation must be compatible with Burn's current Dataset abstraction.

The Dataset must support:

1. Multiple training roots.
2. A single validation root.
3. Independent reading of images, labels, annotations, and label masks for each root.
4. Merging samples across roots.
5. Unique sample IDs.
6. Abnormal-sample handling.
7. Strict / non-strict modes.
8. A dataset inspection command.

Each Dataset item must contain at least:

```text
image_path
label_path
root_id
sample_id
original_width
original_height
yolo_boxes_raw
pixel_boxes_raw
pixel_boxes_after_label_masks
rectangle_polygons
ignore_regions
annotation_meta
source
frame_id
image_data or image_tensor
gt_text
gt_kernel
training_mask
gt_instance
gt_boxes
img_meta
```

Strict mode:

```toml
strict_dataset = true
```

Behavior:

* Missing image: error.
* Missing label: error.
* Invalid label format: error.
* Invalid bbox: error.
* Conflicting critical annotation fields: error.
* Conflicting label_masks schema: error.

Non-strict mode:

```toml
strict_dataset = false
```

Behavior:

* Skip or repair abnormal samples.
* Emit warnings.
* Record issues in the dataset inspection report.
* Do not interrupt training.

The `inspect-dataset` command must be implemented and output:

1. Total sample count.
2. Sample count per root.
3. Empty-label count.
4. Invalid-label count.
5. Number of bboxes deleted by label_masks.
6. Number of bboxes corrected by label_masks.
7. Ignore-region count.
8. Image-size distribution.
9. Bbox-size distribution.
10. Subtitle-position distribution.
11. Abnormal-sample list.

## 8. Preprocessing Design

SubFastNet preprocessing must reference the FAST training-data processing flow instead of a simplified resize + bbox pipeline from common object detection projects.

The core of FAST preprocessing is not simple image scaling. It is built around the following training targets:

```text
gt_instance
gt_text
gt_kernel
training_mask
```

SubFastNet must inherit this idea and adapt it to subtitle bbox data.

Because this project's labels are YOLO bboxes rather than polygons from the original FAST datasets, the preprocessing rule is:

```text
YOLO bbox -> pixel bbox -> rectangle polygon -> gt_instance / gt_text / gt_kernel / training_mask
```

In other words, even though the original annotation is a bbox, target generation should treat the bbox as a four-point rectangle polygon so the FAST mask-based target-generation flow can be reused.

### 8.1 Main Training Preprocessing Flow

Training sample processing order must be:

1. Read image.
2. Decode image.
3. Convert to RGB.
4. Read same-name `labels/*.txt`.
5. Parse YOLO bboxes.
6. Read corresponding metadata from `annotations.jsonl`.
7. Read and apply `label_masks.json`.
8. Convert YOLO normalized bboxes to original-image pixel bboxes.
9. Convert bboxes to four-point rectangle polygons.
10. Apply FAST-style random scale.
11. Synchronously scale image, bboxes, and polygons.
12. Initialize `gt_instance`.
13. Initialize `training_mask`.
14. Draw each valid subtitle bbox into `gt_instance`.
15. Draw ignore regions into `training_mask = 0`.
16. Generate an instance-level kernel source mask for each instance.
17. Generate the merged `gt_kernel` with FAST-style min pooling / pooling kernel.
18. Generate supplementary kernels from shrunken bboxes / polygons.
19. Merge the min-pooled kernel and shrink kernel.
20. Apply FAST-style random horizontal flip.
21. Apply FAST-style random rotate.
22. Apply FAST-style random crop padding.
23. Generate `gt_text` from `gt_instance`.
24. Apply image augmentations such as color jitter / blur.
25. Convert to tensor.
26. Normalize.
27. Batch collate.

The final training batch must contain at least:

```text
imgs
gt_texts
gt_kernels
training_masks
gt_instances
gt_boxes
img_metas
```

Where:

```text
gt_texts       = binary subtitle region mask
gt_kernels     = FAST-style kernel mask
training_masks = loss ignore mask
gt_instances   = instance id mask
gt_boxes       = bboxes after label_masks and geometric transforms
img_metas      = original size, scale ratio, padding, source, frame_id, and other metadata
```

### 8.2 Main Validation Preprocessing Flow

Validation must use deterministic preprocessing.

Validation sample processing order:

1. Read image.
2. Decode image.
3. Convert to RGB.
4. Read YOLO bboxes.
5. Apply `label_masks.json`.
6. Convert YOLO bboxes to pixel bboxes.
7. Convert bboxes to rectangle polygons.
8. Use FAST-style aligned short resize.
9. Record original image size.
10. Record resized size.
11. Synchronously transform bboxes.
12. Generate `gt_text`, `gt_kernel`, `training_mask`, and `gt_instance`.
13. Convert to tensor.
14. Normalize.
15. Batch collate.

Random augmentation is forbidden during validation.

Validation must save:

```text
original_width
original_height
resized_width
resized_height
scale
pad
source
frame_id
```

These fields are used to restore predicted bboxes to original-image pixel coordinates.

### 8.3 Main Inference Preprocessing Flow

Inference does not read labels.

Inference sample processing order:

1. Read image.
2. Decode image.
3. Convert to RGB.
4. Use FAST-style aligned short resize.
5. Record original image size.
6. Record resized size.
7. Convert to tensor.
8. Normalize.
9. Feed the model.
10. Postprocess output bboxes.
11. Restore bboxes to original-image pixel coordinates.

Random augmentation is forbidden during inference.

### 8.4 FAST-Style Aligned Resize

SubFastNet must implement FAST-style `scale_aligned_short`.

Behavior:

1. Scale the image short side according to `short_size`.
2. Preserve aspect ratio.
3. Align the scaled width and height to the model stride or configured alignment.
4. Record the scale ratio.
5. Use the recorded ratio during postprocessing to restore original-image coordinates.

Default:

```toml
short_size = 640
alignment = 32
```

For subtitle detection, non-square input may be allowed, but FAST-style aligned resize semantics must be preserved.

If crop / padding is used to obtain fixed-size training batches, it should only be used during training.

Validation and inference default to aligned short resize.

### 8.5 FAST-Style Random Scale

Training must implement FAST-style random scale.

Default policy:

```toml
scale_min = 0.7
scale_max = 1.3
aspect_min = 0.9
aspect_max = 1.1
```

For subtitle data, the larger FAST Total-Text style scale range may also be kept as an optional experiment:

```toml
scale_min = 0.5
scale_max = 2.0
```

However, the recommended default is the more stable policy:

```toml
scale_min = 0.7
scale_max = 1.3
aspect_min = 0.9
aspect_max = 1.1
```

Random scale must synchronously transform:

```text
image
bbox
rectangle polygon
ignore region
```

### 8.6 FAST-Style Random Crop Padding

Training must implement FAST-style `random_crop_padding`.

It is not ordinary random crop.

Requirements:

1. Crop to a fixed training size.
2. Pad if the image is smaller than the target size.
3. The crop must synchronously affect image, `gt_instance`, `training_mask`, and `gt_kernel`.
4. Valid subtitle regions should still be retained after crop.
5. Bboxes must be re-clipped and validated after crop.
6. Bboxes completely outside the crop are deleted.
7. Bboxes partly inside the crop are clipped to the crop boundary.
8. Bboxes that become too small after crop are deleted or moved to ignore, depending on the strict policy.

Default:

```toml
input_size = 640
```

If a non-square subtitle training size is used, such as 640x384, the same semantics must still be preserved:

```toml
input_width = 640
input_height = 384
```

### 8.7 FAST-Style Random Rotate

Training must implement FAST-style random rotate, but the default angle should be smaller for subtitle scenarios.

FAST uses larger rotation augmentation for general text detection. SubFastNet targets subtitle detection, where subtitles are usually horizontal or nearly horizontal, so the recommended default is:

```toml
random_rotate = true
rotate_angle = 5
```

The maximum is not recommended to exceed:

```toml
rotate_angle = 10
```

Larger angles are allowed only when the training data truly contains obviously tilted subtitles.

Rotation must synchronously affect:

```text
image
gt_instance
training_mask
gt_kernel
bbox
rectangle polygon
ignore region
```

After rotation, the bbox should be regenerated from the polygon's axis-aligned bounding rectangle.

### 8.8 FAST-Style Random Horizontal Flip

Training may implement FAST-style random horizontal flip.

Subtitle detection only cares about regions and does not recognize text content, so horizontal flip will not break OCR labels because this project has no OCR labels.

If OCR is introduced later, horizontal flip must be disabled.

Default:

```toml
random_horizontal_flip = true
flip_prob = 0.5
```

### 8.9 ColorJitter And Blur

Training keeps FAST-style image augmentation.

Default augmentations:

```text
brightness jitter
contrast jitter
saturation jitter
optional gaussian blur
```

Recommended defaults:

```toml
brightness = 0.125
contrast = 0.4
saturation = 0.4
hue = 0.1
gaussian_blur = true
gaussian_blur_prob = 0.5
```

Because subtitles may include strokes, shadows, and compression noise, lightweight jpeg compression noise may be added as a subtitle-specific extension, but it should not replace the original FAST augmentations.

### 8.10 Normalize

SubFastNet defaults to ImageNet mean/std to stay aligned with FAST training practice:

```toml
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
```

If training from scratch without ImageNet / TextNet pretraining, dataset mean/std may also be used through configuration, but default documentation and implementation should first align with FAST.

### 8.11 Preprocessing Operator List

The following operators must be implemented:

```text
DecodeImage
ConvertToRgb
ParseYoloLabel
LoadAnnotationMeta
LoadLabelMasks
ApplyLabelMasks
YoloBoxToPixelBox
BoxToRectanglePolygon
FastRandomScale
FastRandomHorizontalFlip
FastRandomRotate
FastRandomCropPadding
ScaleAlignedShort
DrawInstanceMask
GenerateTrainingMask
GenerateTextMask
GenerateFastKernelMask
MinPoolingKernel
ShrinkKernel
MergeKernel
Normalize
ToTensor
CollateBatch
RestoreBoxToOriginalImage
```

Each operator must describe:

1. Input.
2. Output.
3. Parameters.
4. Whether it is used during training.
5. Whether it is used during validation.
6. Whether it is used during inference.
7. Whether it synchronously transforms bbox / polygon / mask.
8. Error handling under strict / non-strict modes.

## 9. FAST-Style Target Generation

SubFastNet target generation must follow FAST as the main line.

Original FAST training targets include:

```text
gt_texts
gt_kernels
training_masks
gt_instances
```

SubFastNet must preserve these core targets.

### 9.1 From YOLO Bbox To Instance Mask

Because dataset labels are YOLO bboxes, each bbox is a subtitle box.

Processing steps:

1. Read YOLO normalized bboxes.
2. Convert to original-image pixel bboxes.
3. Apply `label_masks.json`.
4. Convert each bbox to a four-point rectangle polygon:

```text
(x1, y1)
(x2, y1)
(x2, y2)
(x1, y2)
```

5. Assign an instance id to each valid bbox:

```text
instance_id = index + 1
```

6. Draw the rectangle polygon into `gt_instance`.

`gt_instance` semantics:

```text
0 = background
1..N = subtitle instance id
```

### 9.2 `training_mask`

`training_mask` defaults to all 1.

The following regions are set to 0:

1. Regions specified as ignore by `label_masks.json`.
2. Bbox regions manually marked as unreliable.
3. Invalid bbox regions.
4. Bbox regions that are too small for stable training.
5. Regions skipped under `strict=false` but that still need masking.

Regions with `training_mask = 0` do not participate in loss.

### 9.3 `gt_text`

`gt_text` is generated from `gt_instance`:

```text
gt_text = gt_instance > 0
```

Semantics:

```text
1 = subtitle text region
0 = background
```

### 9.4 FAST-Style Kernel Mask

SubFastNet's `gt_kernel` must reference FAST kernel generation and should not be reduced to simple bbox shrink only.

Recommended flow:

1. Generate a separate instance kernel source mask for each instance.
2. Use the min-pooling idea to shrink the inside of each instance.
3. Detect overlap regions.
4. Remove overlap regions from the kernel.
5. Shrink bboxes / polygons.
6. Generate the shrink kernel.
7. Merge the min-pooled kernel and shrink kernel.
8. Output the final binary `gt_kernel`.

Goal:

```text
gt_kernel = stable center area of each subtitle instance
```

Kernel purpose:

1. Serve as the postprocessing seed.
2. Separate adjacent subtitle lines.
3. Separate adjacent subtitle boxes.
4. Reduce postprocessing complexity.

### 9.5 `pooling_size`

FAST-style pooling size must be supported.

Default:

```toml
pooling_size = 9
```

Meaning:

1. Controls min-pooling kernel size.
2. Controls kernel shrink strength.
3. Affects separation ability for adjacent instances.
4. Affects preservation of small subtitle boxes.

For subtitle boxes, protection must be added:

```toml
min_kernel_width = 3
min_kernel_height = 3
```

This prevents low-height subtitles from being removed by kernel operations.

### 9.6 `shrink_kernel_scale`

FAST-style shrink kernel scale must be supported.

Default:

```toml
shrink_kernel_scale = 0.1
```

Note: this is not the common bbox shrink ratio 0.5 / 0.7 used in ordinary detection. In FAST, shrink scale is closer to an inward boundary offset ratio, and it works together with min pooling to form the final kernel.

SubFastNet should first use FAST-like defaults, then tune through benchmark / validation.

Recommended initial config:

```toml
pooling_size = 9
shrink_kernel_scale = 0.1
```

### 9.7 Two-Line Subtitle Handling

Two-line subtitles are annotated as one bbox per line.

Therefore, each line is a separate instance:

```text
line_1 -> instance 1
line_2 -> instance 2
```

Do not merge two-line subtitles.

Do not combine two lines into one bbox.

Kernel generation must ensure that upper and lower subtitle lines can be separated.

### 9.8 Target Output

Final training sample output:

```text
imgs: [3, H, W]
gt_texts: [H, W]
gt_kernels: [H, W]
training_masks: [H, W]
gt_instances: [H, W]
gt_boxes: Vec<Box>
img_metas: ImageMeta
```

After batching:

```text
imgs: [B, 3, H, W]
gt_texts: [B, H, W]
gt_kernels: [B, H, W]
training_masks: [B, H, W]
gt_instances: [B, H, W]
gt_boxes: Vec<Vec<Box>>
img_metas: Vec<ImageMeta>
```

## 10. Default SubFastNet Training Preprocessing Config

The default config should primarily follow the FAST style:

```toml
input_size = 640
short_size = 640
alignment = 32

pooling_size = 9
shrink_kernel_scale = 0.1
min_kernel_width = 3
min_kernel_height = 3

random_scale = true
scale_min = 0.7
scale_max = 1.3
aspect_min = 0.9
aspect_max = 1.1

random_horizontal_flip = true
flip_prob = 0.5

random_rotate = true
rotate_angle = 5

random_crop_padding = true

color_jitter = true
brightness = 0.125
contrast = 0.4
saturation = 0.4
hue = 0.1

gaussian_blur = true
gaussian_blur_prob = 0.5

normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
```

If non-square input is used for subtitle scenarios, FAST-style crop padding and aligned resize semantics must still be preserved:

```toml
input_width = 640
input_height = 384
short_size = 384
alignment = 32
```

Do not degrade preprocessing into an ordinary YOLO detection pipeline.

Do not generate only bbox targets.

Do not skip `gt_instance`.

Do not skip `gt_kernel`.

Do not skip `training_mask`.

Do not ignore `label_masks.json`.

## 11. Difference Boundaries With FAST

SubFastNet is allowed to differ from FAST in these ways:

1. The original label source is YOLO bbox, not polygon.
2. Bboxes are converted to rectangle polygons before entering the FAST-style target pipeline.
3. Output is axis-aligned pixel bbox, not polygon.
4. Subtitle scenarios use a smaller default rotation angle.
5. Subtitle scenarios do not default to arbitrary-shape text augmentation.
6. Subtitle scenarios do not perform OCR.
7. Subtitle scenarios do not perform cross-language text recognition.
8. Postprocessing only needs to produce bboxes and does not need to restore complex text polygons.

SubFastNet should not differ from FAST in these ways:

1. It should not remove `gt_instance`.
2. It should not remove `gt_text`.
3. It should not remove `gt_kernel`.
4. It should not remove `training_mask`.
5. It should not simplify kernel generation to ordinary bbox shrink only.
6. It should not replace training preprocessing with ordinary YOLO detection augmentation.
7. It should not ignore FAST-style scale / crop padding / aligned resize.
8. It should not ignore `pooling_size`.
9. It should not ignore `shrink_kernel_scale`.
10. It should not ignore the min-pooling / overlap-suppression kernel idea.

## 12. Model Architecture: SubFastNet

The model name must be:

```text
SubFastNet
```

The default architecture must reference FAST's TextNet.

Do not default to:

* MobileNet
* MobileNetV2
* MobileNetV3
* LCNet
* ShuffleNet
* YOLO backbone
* EfficientNet
* ResNet

These may be used at most as baselines or ablations, not as the main architecture.

Main architecture:

```text
SubFastNet = TextNet-style backbone + lightweight feature fusion + FAST-like minimalist kernel detection head
```

TextNet is the backbone used by FAST for text detection tasks. The agent should first reference the TextNet-T / TextNet-S / TextNet-B structures from the FAST paper and official implementation. If the project already has a local FAST / TextNet implementation or notes, read and align with them first.

Model variants:

```text
SubFastNet-Tiny
SubFastNet-Small
SubFastNet-Base
```

Default first choice:

```text
SubFastNet-Tiny
```

Default usage:

* Tiny: highest FPS.
* Small: speed / accuracy balance.
* Base: optional high-accuracy experiment, not the default.

TextNet-style backbone requirements:

1. Target text detection, not classification-transfer first.
2. Output multi-scale features.
3. Preserve localization ability for long, thin subtitle regions.
4. Support stride 4 / stride 8 / stride 16 features.
5. Stride 32 features are optional only in Small / Base.
6. Prefer operators stably supported by Burn CPU / CUDA / WGPU.
7. Do not depend on custom CUDA kernels.
8. Do not depend on complex dynamic shapes.
9. Do not introduce an OCR branch.

Recommended feature levels:

```text
P2: stride 4, optional, for low-height subtitles
P3: stride 8, main detection feature
P4: stride 16, semantic enhancement
P5: stride 32, only optional for Small/Base
```

Lightweight feature fusion:

1. 1x1 conv to align channels.
2. Nearest upsample.
3. Add or concat fusion.
4. Lightweight conv refinement.
5. Output a unified detection feature map.

Do not use heavy FPN / PAN.

Do not introduce a complex polygon branch for generalizing to natural-scene text.

## 13. Detection Head

The detection head uses a FAST-like minimalist kernel representation.

Default output has two logits:

```text
text_region_logits: [B, 1, H, W]
kernel_logits: [B, 1, H, W]
```

Region head:

* Predicts the complete subtitle region.
* Supervision comes from `gt_text`.

Kernel head:

* Predicts the subtitle kernel after shrink / pooling.
* Supervision comes from `gt_kernel`.
* Used during postprocessing to separate adjacent subtitle boxes.

The bbox regression head is disabled by default.

If the bboxes obtained by region / kernel postprocessing are not precise enough, a lightweight bbox refinement head may be added, but this is not the default main line.

Do not add an OCR head by default.

Do not add a text recognition head by default.

Do not add a character classification head by default.

## 14. Loss Design

Default loss:

```text
total_loss = region_loss * region_weight
           + kernel_loss * kernel_weight
           + optional_bbox_loss * bbox_weight
```

Region loss:

```text
BCEWithLogits + Dice Loss
```

Kernel loss:

```text
BCEWithLogits + Dice Loss
```

Bbox loss:

Disabled by default.

If the bbox refinement head is enabled, use:

```text
SmoothL1
IoU loss
GIoU loss
```

Ignore mask:

1. Region loss must support `training_mask`.
2. Kernel loss must support `training_mask`.
3. Ignore regions do not participate in positive / negative sample statistics.
4. Ignore regions produced by `label_masks.json` must take effect.

Optional OHEM:

* Used to control excessive background regions.
* May be disabled by default.
* If enabled, it should apply only to negative regions for region loss.

Loss breakdown must be output:

```text
total_loss
region_bce_loss
region_dice_loss
kernel_bce_loss
kernel_dice_loss
bbox_loss
ignored_area_ratio
positive_region_ratio
positive_kernel_ratio
```

Do not implement:

```text
CTC loss
attention decoder loss
CER
WER
text exact match
```

## 15. Postprocessing Design

Postprocessing must be simple, fast, and suitable for high-FPS desktop inference.

Default flow:

1. Apply sigmoid to region logits.
2. Apply sigmoid to kernel logits.
3. Apply region threshold.
4. Apply kernel threshold.
5. Use kernel as seeds.
6. Perform lightweight aggregation inside the region.
7. Use connected components or an equivalent fast aggregation.
8. Generate an axis-aligned bbox for each component.
9. Compute confidence.
10. Filter bboxes that are too small.
11. Optional NMS.
12. Restore to original-image pixel coordinates.
13. Output `x1, y1, x2, y2, confidence`.

Output bboxes must be in pixel coordinates.

Do not output polygons by default.

Do not restore complex curved text by default.

Do not perform cross-frame subtitle merging by default.

Bbox restore must account for:

1. Resize scale.
2. Letterbox padding.
3. Original image width and height.
4. Coordinate clipping.
5. Minimum bbox-size filtering.

Recommended confidence:

```text
confidence = mean(region_prob within component)
```

Or:

```text
confidence = weighted_mean(region_prob, kernel_prob)
```

Postprocessing time must be benchmarked.

Postprocessing should output:

```text
postprocess_latency_ms
candidate_count
final_box_count
```

## 16. Training Flow

The training flow must include:

1. Read config file.
2. Initialize backend: CPU / CUDA / WGPU.
3. Set seed.
4. Load multiple train roots.
5. Load single val root.
6. Build Burn Dataset.
7. Build dataloader.
8. Build SubFastNet.
9. Initialize optimizer.
10. Initialize scheduler.
11. Integrate Burn learner or a standard Burn training loop.
12. Integrate Burn TUI.
13. Validate at configured intervals.
14. Save checkpoints.
15. Support resume.
16. Save best model.
17. Save final model.
18. Output metrics.
19. Output training summary.
20. Output error-sample analysis.

Training must output:

```text
epoch
step
learning_rate
total_loss
region_loss
kernel_loss
bbox_loss
samples_per_second
batch_time
data_time
gpu_memory or backend memory information
```

## 17. Validation Flow

The validation set has only one root.

Random augmentation is forbidden during validation.

Validation flow:

1. Load val root.
2. Apply deterministic preprocessing consistent with training.
3. Batch inference.
4. Compute validation loss.
5. Execute postprocessing.
6. Restore predicted bboxes to original-image pixel coordinates.
7. Convert YOLO labels to ground-truth pixel bboxes.
8. Apply `label_masks.json` before using them as GT.
9. Compute bbox IoU.
10. Match TP / FP / FN by IoU threshold.
11. Compute precision.
12. Compute recall.
13. Compute F1.
14. Compute mean IoU.
15. Compute FPS.
16. Compute p50 / p95 latency.
17. Save error-sample analysis.

Validation metrics:

```text
val_loss
precision
recall
f1
mean_iou
fps
latency_p50
latency_p95
false_positive_count
false_negative_count
ignored_sample_count
```

Error-sample analysis output:

```text
false_positive.jsonl
false_negative.jsonl
low_iou.jsonl
ignored_samples.jsonl
```

Each error record should contain:

```json
{
  "image": "path/to/image.jpg",
  "source": "optional",
  "frame_id": "optional",
  "width": 1920,
  "height": 1080,
  "pred_boxes": [],
  "gt_boxes": [],
  "iou": [],
  "reason": "false_positive | false_negative | low_iou | ignored"
}
```

## 18. Metrics Design

Training metrics:

```text
total_loss
region_loss
kernel_loss
bbox_loss
learning_rate
samples_per_second
batch_time
data_time
positive_region_ratio
positive_kernel_ratio
ignored_area_ratio
```

Validation metrics:

```text
val_loss
precision
recall
f1
mean_iou
fps
latency_p50
latency_p95
postprocess_latency
false_positive_count
false_negative_count
ignored_sample_count
```

Benchmark metrics:

```text
dataloader_throughput
preprocess_throughput
train_step_time
validation_step_time
inference_fps
latency_p50
latency_p95
postprocess_latency
memory_usage
```

All metrics must be output at least to:

```text
console
Burn TUI
metrics.jsonl
summary.json
```

## 19. Burn TUI Requirements

Standardized Burn TUI must be integrated.

The TUI must show at least:

```text
epoch
step
progress
total_loss
region_loss
kernel_loss
val_loss
precision
recall
f1
mean_iou
learning_rate
samples_per_second
batch_time
data_time
fps
latency
checkpoint status
```

If the backend can provide memory information, also show:

```text
memory usage
gpu memory
```

The TUI should not break ordinary log output.

It must support disabling:

```toml
tui_enabled = false
```

## 20. Backend Support

The project must support:

```text
cpu
cuda
wgpu
```

Config example:

```toml
backend = "wgpu"
```

Requirements:

1. CPU can train.
2. CUDA can train.
3. WGPU can train.
4. Reuse the same Dataset / preprocess / model / loss / metrics as much as possible.
5. Backend differences should only be in initialization and feature gates.
6. Do not depend on custom kernels available only on CUDA.
7. Postprocessing may default to CPU, but its time must be measured.
8. If GPU postprocessing is implemented, a CPU fallback must be retained.

## 21. Config File Requirements

The config file should stay necessary and concise.

Do not design an overly complex schema.

Do not hardcode training parameters.

Minimum config fields:

```toml
experiment_name = "subfastnet_tiny"
output_dir = "outputs/subfastnet_tiny"
seed = 42

backend = "wgpu"

train_roots = [
  "/data/subtitle/train_a",
  "/data/subtitle/train_b"
]

val_root = "/data/subtitle/val"

model_variant = "tiny"

input_size = 640
short_size = 640
alignment = 32

batch_size = 16
epochs = 100
learning_rate = 0.001

validation_interval = 1
checkpoint_interval = 1
log_interval = 50

threshold_region = 0.5
threshold_kernel = 0.5
iou_threshold = 0.5

pooling_size = 9
shrink_kernel_scale = 0.1
min_kernel_width = 3
min_kernel_height = 3

augment_enabled = true
strict_dataset = false
tui_enabled = true

resume = ""
```

The agent may naturally add a small number of fields as needed by the implementation, but should not generate dozens of unnecessary config items.

## 22. CLI Requirements

A command-line interface must be provided.

Commands must include at least:

```text
train
validate
infer
inspect-dataset
benchmark
```

Examples:

```bash
cargo run --release -- train --config configs/train.toml
```

```bash
cargo run --release -- validate --config configs/train.toml --checkpoint outputs/subfastnet_tiny/best
```

```bash
cargo run --release -- infer --config configs/train.toml --checkpoint outputs/subfastnet_tiny/best --image sample.jpg
```

```bash
cargo run --release -- inspect-dataset --config configs/train.toml
```

```bash
cargo run --release -- benchmark --config configs/train.toml --checkpoint outputs/subfastnet_tiny/best
```

The `infer` command outputs pixel-coordinate bboxes.

## 23. Benchmark Requirements

Benchmark must be implemented.

Benchmark contents:

1. Dataloader throughput.
2. Preprocess throughput.
3. Train step time.
4. Validation step time.
5. Inference FPS.
6. Latency p50.
7. Latency p95.
8. Postprocess latency.
9. Memory usage.
10. End-to-end latency.

Benchmark output:

```text
console
Burn TUI
metrics.jsonl
summary.json
```

Benchmark must distinguish:

```text
preprocess time
model forward time
postprocess time
end-to-end time
```

The FPS target is based on end-to-end inference, not only model forward.

## 24. Checkpoint And Resume

The project must support:

1. Periodic checkpoint saving.
2. Saving the best model.
3. Saving the final model.
4. Resume from checkpoint.
5. Saving optimizer state.
6. Saving scheduler state.
7. Saving epoch / step.
8. Saving config snapshot.
9. Saving metrics summary.

Suggested output directory:

```text
outputs/
`-- subfastnet_tiny/
    |-- checkpoints/
    |-- best/
    |-- final/
    |-- metrics.jsonl
    |-- summary.json
    |-- config.snapshot.toml
    `-- errors/
```

## 25. Documentation Requirements

Detailed documentation must be generated, but it should not over-expand the internals of every `.rs` file.

At least generate the root documents:

```text
README.md
DESIGN.md
```

Keep the topic documents under `docs/`:

```text
docs/DATASET.md
docs/PREPROCESSING.md
docs/MODEL.md
docs/TRAINING.md
docs/METRICS.md
docs/INFERENCE.md
docs/BENCHMARK.md
```

The documentation must explain:

1. SubFastNet is a subtitle region detection model.
2. SubFastNet is not OCR.
3. Dataset structure.
4. YOLO label parsing.
5. Multi-train-root loading.
6. Single-val-root loading.
7. How `label_masks.json` participates in training-sample injection.
8. How `annotations.jsonl` is used as metadata.
9. FAST-style preprocessing flow.
10. `gt_instance` generation.
11. `gt_text` generation.
12. `gt_kernel` generation.
13. `training_mask` generation.
14. TextNet-style backbone.
15. FAST-like minimalist kernel head.
16. Loss design.
17. Bbox metric calculation.
18. High-FPS inference optimization points.
19. CPU / CUDA / WGPU backend notes.
20. Burn TUI.
21. Checkpoint / resume.
22. Benchmark method.
23. Error-sample analysis.

## 26. Test Requirements

Tests must be included.

Tests must cover at least:

1. YOLO label parser.
2. Bbox normalized -> pixel conversion.
3. Bbox -> rectangle polygon conversion.
4. Resize / aligned short resize bbox synchronous transform.
5. Random scale bbox / polygon synchronous transform.
6. Random rotate bbox / polygon synchronous transform.
7. Random crop padding mask synchronous transform.
8. label_masks application.
9. annotations.jsonl reading.
10. gt_instance generation.
11. gt_text generation.
12. gt_kernel generation.
13. training_mask generation.
14. Dataset multi-root merge.
15. Empty-label handling.
16. Strict / non-strict behavior.
17. Postprocess bbox restore.
18. Metric IoU matching.
19. Smoke train.

Smoke train should run through with a tiny dataset:

```text
dataset load
batcher
forward
loss
backward
optimizer step
validation
checkpoint save
```

## 27. Engineering Module Responsibilities

A standard Rust single crate or workspace may be used.

Do not over-list the details of every `.rs` file in the design document, but the following module responsibilities must be included:

```text
config
dataset
preprocess
target
model
loss
metrics
train
validate
infer
postprocess
benchmark
checkpoint
tui
utils
```

Responsibility descriptions:

config:

* Read TOML config.
* Validate required fields.
* Save config snapshot.

dataset:

* Load multiple train roots.
* Load a single val root.
* Parse images / labels / annotations / label_masks.
* Implement a Burn Dataset-compatible structure.

preprocess:

* Image decoding.
* FAST-style random scale.
* FAST-style random crop padding.
* FAST-style random rotate.
* FAST-style horizontal flip.
* Aligned short resize.
* Normalize.
* Synchronous bbox / polygon / mask transforms.
* Tensor conversion.

target:

* Generate gt_instance.
* Generate gt_text.
* Generate gt_kernel.
* Generate training_mask.
* Generate gt_boxes.

model:

* Implement SubFastNet.
* Implement TextNet-style backbone.
* Implement lightweight feature fusion.
* Implement FAST-like region/kernel head.

loss:

* Region loss.
* Kernel loss.
* Optional bbox loss.
* `training_mask` ignore support.

metrics:

* Bbox IoU.
* Precision / recall / F1.
* Mean IoU.
* Latency / FPS.

train:

* Training loop.
* Optimizer.
* Scheduler.
* Burn TUI.
* Checkpoint.

validate:

* Validation loop.
* Error-sample analysis.

infer:

* Single-image inference.
* Batch inference.
* Output pixel-coordinate bboxes.

postprocess:

* Sigmoid.
* Threshold.
* Component aggregation.
* Bbox restore.
* NMS or lightweight filtering.

benchmark:

* Preprocess.
* Forward.
* Postprocess.
* End-to-end FPS.

## 28. Forbidden Items

Do not default to Python.

Do not default to PyTorch.

Do not turn the task into OCR.

Do not implement a character recognition head.

Do not implement CTC.

Do not implement an attention decoder.

Do not output CER / WER.

Do not ask again whether the task is OCR.

Do not ask again whether bboxes are subtitle boxes.

Do not ask again whether this is single-class.

Do not ask again how two-line subtitles are annotated.

Do not assume the training set is a single directory.

Do not assume the validation set has multiple directories.

Do not ignore `label_masks.json`.

Do not ignore `annotations.jsonl`.

Do not over-design the config file.

Do not hardcode training parameters.

Do not default to MobileNetV3.

Do not default to LCNet.

Do not default to ShuffleNet.

Do not default to a YOLO backbone.

Do not default to arbitrary curved text or complex polygon support.

Do not sacrifice FPS with complex postprocessing.

Do not do cross-frame subtitle merging by default.

Do not degrade FAST-style preprocessing into ordinary YOLO detection preprocessing.

Do not skip `gt_instance`.

Do not skip `gt_text`.

Do not skip `gt_kernel`.

Do not skip `training_mask`.

Do not write documentation that contains only concepts without actionable engineering design.

## 29. Questions That May Be Confirmed

Only the following situations allow asking the user for confirmation:

1. The real schema of `label_masks.json` cannot be inferred from files.
2. `label_masks.json` contains mutually conflicting rules.
3. Fields in `annotations.jsonl` would change label semantics.
4. The concrete FAST / TextNet structure cannot be determined from the paper, official implementation, or project files.
5. An implementation decision would directly change training targets.
6. An implementation decision would directly change loss.
7. An implementation decision would directly change inference output semantics.

Other ordinary engineering questions should not be pushed back to the user.

## 30. Final Deliverables

The final delivery must include:

1. Complete project code.
2. Concise config file.
3. README.
4. Detailed design document.
5. Dataset implementation.
6. Multi-train-root loading.
7. Single-val-root loading.
8. YOLO bbox parser.
9. `label_masks.json` application logic.
10. `annotations.jsonl` metadata reading.
11. FAST-style preprocessing operators.
12. Target generation.
13. SubFastNet model.
14. TextNet-style backbone.
15. FAST-like region/kernel heads.
16. Loss.
17. Metrics.
18. Burn TUI training.
19. Validation.
20. Inference.
21. Benchmark.
22. Checkpoint / resume.
23. Tests.
24. Error-sample analysis output.
25. Summary and metrics output.

The final project must pass the following basic flow:

```bash
cargo run --release -- inspect-dataset --config configs/train.toml
cargo run --release -- train --config configs/train.toml
cargo run --release -- validate --config configs/train.toml --checkpoint outputs/subfastnet_tiny/best
cargo run --release -- infer --config configs/train.toml --checkpoint outputs/subfastnet_tiny/best --image sample.jpg
cargo run --release -- benchmark --config configs/train.toml --checkpoint outputs/subfastnet_tiny/best
```

## 31. Core Acceptance Criteria

Whether the project is acceptable is judged by:

1. Whether it is truly subtitle region detection rather than OCR.
2. Whether it correctly supports multiple training roots.
3. Whether it correctly supports a single validation root.
4. Whether it correctly parses YOLO bboxes.
5. Whether it forcibly applies `label_masks.json`.
6. Whether it reads and uses `annotations.jsonl`.
7. Whether it uses the SubFastNet name.
8. Whether TextNet / FAST-like is the default architecture.
9. Whether FAST-style preprocessing is used.
10. Whether `gt_instance`, `gt_text`, `gt_kernel`, and `training_mask` are generated.
11. Whether pixel-coordinate bboxes are output.
12. Whether CPU / CUDA / WGPU are supported.
13. Whether Burn TUI is integrated.
14. Whether validation-set metrics exist.
15. Whether FPS / latency benchmark exists.
16. Whether checkpoint / resume exists.
17. Whether complete documentation exists.
18. Whether the requirements are not incorrectly expanded into OCR or general text detection.
