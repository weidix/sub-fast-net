# Benchmark

Benchmark measures:

```text
dataloader throughput
preprocess throughput
train-step time status
validation-step time
inference FPS
latency p50
latency p95
postprocess latency
end-to-end latency
memory usage note
candidate_count
final_box_count
```

It distinguishes:

```text
preprocess time
forward time
postprocess time
end-to-end time
```

The benchmark runs multiple validation samples, capped by `max_val_samples` when configured or 32 by default. It reports p50/p95 for end-to-end, preprocess, forward, and postprocess stages.

`train_step_time` is measured with the configured autodiff backend by executing one real training step: preprocess a batch, forward through `SubFastNet`, compute region/kernel tensor loss, run backward, and apply one Adam optimizer update. The field is never filled with forward-only timing.

`memory_usage` reports Burn's CPU RAM system metric when available. Backend/GPU allocator memory remains explicitly unsupported because Burn 0.21 does not expose a stable public memory-usage method on the generic `Backend` trait for this code path.

Run:

```bash
cargo run --release -- benchmark --config configs/train.toml --checkpoint outputs/subfastnet_tiny/best
```

Outputs are written to the DESIGN-standard files:

```text
outputs/<experiment>/summary.json
outputs/<experiment>/metrics.jsonl
```

Benchmark records include `record_type = "benchmark"` and are appended to `metrics.jsonl`. `summary.json` keeps any existing training summary and adds or replaces a top-level `benchmark` object. The command also keeps the extra dedicated artifacts `benchmark_summary.json` and `benchmark_metrics.jsonl`.

The same metrics can be shown through Burn TUI during training/validation. The standalone benchmark command emits the same fields to console and files; it does not fabricate GPU memory when Burn does not expose it.
