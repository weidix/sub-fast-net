# Preprocessing

Training follows the FAST-style flow adapted to YOLO subtitle bboxes:

```text
decode RGB image
parse YOLO labels
read annotations
apply label_masks
YOLO bbox -> pixel bbox -> rectangle polygon
FastRandomScale
FastRandomHorizontalFlip
FastRandomRotate
FastRandomCropPadding
ColorJitter / Gaussian blur
Normalize
ToTensor
CollateBatch
```

Geometry transforms are synchronous for image, boxes, rectangle polygons, and ignore regions. Rotation updates the polygon first and regenerates the axis-aligned bbox from the polygon bounds. Crop/padding clips polygons and deletes polygons that fall fully outside the crop.

Targets are produced after the final crop/pad step from rectangle polygon semantics:

```text
rectangle polygon -> gt_instance -> gt_text
rectangle polygon + min pooling + shrink kernel -> gt_kernel
ignore regions -> training_mask = 0
```

Even though current labels are YOLO boxes, target generation does not bypass the FAST-style polygon/mask route.

Validation and inference are deterministic and use `ScaleAlignedShort`: scale the short side to `short_size`, preserve aspect ratio, align width and height to `alignment`, record scale and padding, and restore output boxes back to original image coordinates.

Batch output contains:

```text
imgs
gt_texts
gt_kernels
training_masks
gt_instances
gt_boxes
img_metas
```

`img_metas` records original size, resized size, scale, padding, source, and frame id.
