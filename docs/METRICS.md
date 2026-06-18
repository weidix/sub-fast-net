# Metrics

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

Detection matching uses axis-aligned bbox IoU and the configured `iou_threshold`.

Validation precision, recall, and F1 are computed from global TP/FP/FN counts over the validation run, not by averaging per-sample precision/recall/F1. `mean_iou` is the mean IoU over matched true positives. `latency_p50` and `latency_p95` are forward plus postprocess latency; `postprocess_latency` is measured separately so postprocess cost is visible.

`ignored_sample_count` includes samples that cannot be loaded/preprocessed and samples marked by `label_masks.json` with `drop_image`.

Validation writes error analysis JSONL:

```text
errors/false_positive.jsonl
errors/false_negative.jsonl
errors/low_iou.jsonl
errors/ignored_samples.jsonl
```
