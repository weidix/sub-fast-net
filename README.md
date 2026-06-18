# SubFastNet

SubFastNet is a Rust/Burn subtitle region detector. It detects subtitle boxes in image frames or ROI crops and outputs pixel-coordinate bounding boxes with confidence. It is not OCR: there is no recognition head, CTC, attention decoder, text decoding, translation, or subtitle parsing.

## Commands

```bash
cargo run --release -- inspect-dataset --config configs/train_wgpu.toml
cargo run --release -- train --config configs/train_wgpu.toml
cargo run --release -- validate --config configs/train_wgpu.toml --checkpoint outputs/subfastnet_tiny_wgpu/best
cargo run --release -- infer --config configs/train_wgpu.toml --checkpoint outputs/subfastnet_tiny_wgpu/best --image sample.jpg
cargo run --release -- benchmark --config configs/train_wgpu.toml --checkpoint outputs/subfastnet_tiny_wgpu/best
```

For a short CPU verification run:

```bash
cargo run --release -- train --config configs/smoke.toml
```

The CUDA/BF16 training configuration is `configs/train_cuda.toml`.

## Project Shape

The crate is organized around `config`, `dataset`, `preprocess`, `target`, `model`, `loss`, `metrics`, `train`, `validate`, `infer`, `postprocess`, `benchmark`, and `checkpoint`.

The default architecture is `SubFastNet-Tiny`: a compact TextNet-style, FAST-like model with stride 4/8/16 features, lightweight fusion, and separate text-region/kernel heads. Training summaries include both a parameter-based size estimate and the actual saved `final/model.bin` artifact size when a checkpoint is written; the actual artifact size is the field to use for the 1-4 MB target check.

## Documentation

- [Dataset](docs/DATASET.md)
- [Preprocessing](docs/PREPROCESSING.md)
- [Model](docs/MODEL.md)
- [Training](docs/TRAINING.md)
- [Metrics](docs/METRICS.md)
- [Inference](docs/INFERENCE.md)
- [Benchmark](docs/BENCHMARK.md)

## Verification

Use:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```
