# Dataset

Each dataset root must contain:

```text
images/*.jpg
labels/*.txt
annotations.jsonl
label_masks.json
```

`train_roots` supports multiple roots. `val_root` is a single root. The default `configs/train_wgpu.toml` does not cap training samples, so all configured training roots participate in full training. Smoke limits belong in `configs/smoke.toml`.

If `max_train_samples` is configured for a smoke or debug run, samples are selected by round-robin across roots instead of truncating the globally sorted list, so a small cap does not silently use only root 0.

`SubtitleDataset` implements Burn 0.21's `burn_dataset::Dataset<DatasetSample>` trait, while the current training loop still uses explicit indexed loading to preserve the custom FAST-style preprocessing and ignored-sample handling.

## Labels

`labels/*.txt` uses YOLO bbox rows:

```text
class_id x_center y_center width height
```

All boxes are subtitle regions. Empty labels mean no subtitle boxes unless `strict_dataset = true` requires an error for a missing file.

## label_masks.json

`label_masks.json` is applied after raw YOLO labels and before target generation. Supported actions include dropping an image, deleting a bbox, correcting a bbox, adding a bbox, adding an ignore region, excluding a region from loss, and marking labels unreliable.

Existing schema is supported:

```json
{"items": {"sample_id": {"0": {"masked": true}}}}
```

Additional compatible fields are `drop_image`, `deleted`, `bbox`, `add_bbox`, `ignore_region`, `unreliable`, `exclude_from_loss`, and `action`.

`drop_image` marks the sample as ignored. In non-strict mode it is recorded in inspection and validation ignored reports; in strict mode sample loading errors. It is not treated as a normal background-only training image. `ignore_region`, `unreliable`, and `exclude_from_loss` remain loss-mask regions and set `training_mask = 0`.

## annotations.jsonl

Annotations are read as metadata and validation hints. They validate image size and provide `source_video`, `frame_index`, ROI offsets, filter regions, and detection metadata for reports. They do not override training labels unless an explicit future schema says so.

## Inspection

```bash
cargo run --release -- inspect-dataset --config configs/train_wgpu.toml
```

The report includes sample counts, empty/invalid labels, deleted/corrected/added masks, ignore regions, dropped images, image-size distribution, bbox-size distribution, subtitle-position distribution, and abnormal samples.
