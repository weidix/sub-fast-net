# Training

Training loads `TrainConfig`, initializes CPU/CUDA/WGPU through Burn features, builds multi-root training and single-root validation datasets, runs SubFastNet, computes masked region/kernel loss, validates at intervals, and saves checkpoints.

Default loss:

```text
region BCE + region Dice + kernel BCE + kernel Dice
```

`training_mask = 0` excludes ignored or unreliable regions from loss.

Checkpoints are written to:

```text
outputs/<experiment>/checkpoints/
outputs/<experiment>/best/
outputs/<experiment>/final/
```

Each checkpoint saves model weights, Adam optimizer state, scheduler state, epoch, step, best F1, learning rate, config snapshot, and metrics summary. `resume` loads the model, optimizer state, scheduler epoch, step, and best metrics.

`tui_enabled = true` now routes training and validation metrics through Burn 0.21's standard `TuiMetricsRendererWrapper` when stdout is an interactive terminal. The custom detector loop still owns FAST-style targets, checkpoint metadata, validation error JSONL, and postprocess metrics, but it sends Burn renderer metric/progress events instead of printing a fake TUI.

The TUI registers epoch/step/progress through Burn's renderer progress metadata and shows:

```text
total_loss
region_loss
kernel_loss
bbox_loss
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
latency_p50
latency_p95
postprocess_latency
memory_usage_gb
checkpoint status
```

`memory_usage_gb` is Burn's CPU/system RAM metric. GPU allocator memory is not invented because the generic Burn 0.21 `Backend` path used here does not expose a stable backend memory API.

When stdout is not a terminal, Burn's TUI renderer cannot take over the alternate screen, so training falls back to ordinary compact console logs while metrics are still written to `metrics.jsonl`.

`drop_image` records from `label_masks.json` are skipped during training. Non-strict dataset mode records them as ignored samples; strict mode errors at sample load time. `ignore_region`, `unreliable`, and `exclude_from_loss` continue to become `training_mask = 0` regions.

## CUDA Mixed Precision

`mixed_precision` is an opt-in CUDA training setting:

```toml
backend = "cuda"
mixed_precision = "off"  # "off" | "bf16" | "fp16"
```

The default is `"off"`, which keeps the existing `Cuda<f32, i32>` training path. `"bf16"` uses `Cuda<bf16, i32>` for model forward/backward. `"fp16"` uses `Cuda<f16, i32>` plus dynamic loss scaling.

The loss formula is unchanged. Loss inputs are cast to FP32 before masked BCE, Dice, ratio, and scalar reductions so sigmoid/log/division/reduction and finite checks run in FP32. Target tensor construction and dataloader behavior are unchanged.

FP16 training multiplies `total_loss` by an initial scale of `1024`, unscales gradients before Adam, skips the optimizer step when loss or gradients contain NaN/Inf, halves the scale on overflow, and grows it after sustained finite steps. BF16 uses the same finite loss guard without loss scaling.

Burn 0.21's Adam state follows the selected backend precision; this project does not currently add a separate FP32 master-weight optimizer wrapper. Checkpoints are still recorded with Burn's full-precision recorder.

Smoke check:

```powershell
cargo check --features backend-cuda
cargo test mixed_precision_defaults_to_off_and_requires_cuda --features backend-cuda
```

Runtime NaN/Inf validation is visible through `mixed_precision_skip_update` console lines and `metrics.jsonl`: `total_loss` should stay finite. For profiling, compare warm averages in `training_profile_summary.json`, especially `forward_*_time`, `loss_compute_*_time`, `backward_*_time`, `optimizer_step_*_time`, and `batch_time`; ignore the cold first step.
