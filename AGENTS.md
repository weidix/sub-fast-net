# AGENTS.md

## Project Background

`SubFastNet` is a Rust project for building a lightweight DB-style subtitle detector based on Burn. The model detects subtitle regions from video frames or subtitle ROI crops and outputs geometry only: bounding boxes and/or masks. It does not perform OCR or text recognition.

## Project Goals

1. Implement a lightweight subtitle detector with Burn.
2. Support detection from full video frames and cropped subtitle ROI inputs.
3. Output `bbox` and/or `mask`; do not add text recognition.
4. Train a dedicated subtitle model with real video frames plus synthetic subtitle data.
5. Keep model size in the 1-4 MB range.
6. Support low-latency inference on edge devices.
7. In subtitle scenes, aim for detection quality close to or better than general OCR detection models.

## Scope Rules

- This project is a detector, not an OCR system.
- Do not add recognition, language modeling, text decoding, subtitle parsing, or translation features unless explicitly requested.
- Prefer small, purpose-built model and data pipelines over general OCR-detector complexity.
- Keep implementation choices compatible with Burn and Rust-first deployment.
- Optimize for subtitle-specific accuracy, compact model size, and predictable inference latency.

## Expected Architecture Direction

- `dataset`: load real frames, ROI crops, labels, and synthetic subtitle samples.
- `augment`: subtitle-specific synthesis and augmentation such as font, stroke, shadow, blur, compression, scaling, contrast, and background variation.
- `model`: compact Burn detector backbone and heads for bbox and/or mask prediction.
- `train`: training loop, loss calculation, checkpointing, metrics, and export.
- `infer`: low-latency inference API for frames or ROI images.
- `eval`: subtitle detection metrics, latency measurement, and comparison against baseline OCR detection results.

Use these module names as guidance, not as mandatory structure if the actual codebase evolves differently.

## Model Constraints

- Target serialized model size: 1-4 MB.
- Prefer compact CNN-style or similarly lightweight architectures.
- Avoid large transformer-style models unless there is a strong measured reason.
- Keep input resolution and feature width aligned with subtitle detection latency requirements.
- Make bbox/mask heads explicit and keep recognition-free outputs.

## Data And Training Guidelines

- Use real subtitle frames as the quality anchor.
- Use synthetic subtitle overlays to improve coverage across fonts, colors, outlines, shadows, blur, compression artifacts, aspect ratios, and backgrounds.
- Keep labels focused on subtitle regions, not individual characters.
- Track real-data and synthetic-data performance separately where practical.
- Include hard negatives: UI text, watermarks, signs, captions outside the subtitle region, and text-like background patterns.

## Inference Guidelines

- Support both full-frame detection and ROI-based detection.
- Keep preprocessing deterministic and documented.
- Output coordinates in a clearly defined image space.
- Prefer APIs that can be reused by CLI, tests, and future embedded integrations.
- Measure latency with realistic frame sizes and edge-device-oriented settings.

## Verification

Before claiming detector behavior is complete, verify with:

- `cargo fmt --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test`
- Targeted dataset/model tests when those modules exist.
- Latency and model-size checks when inference/export code exists.

If Burn or native dependencies are not yet available, state the exact command that failed and the reason.

## Development Style

- Keep changes narrow and directly tied to the current request.
- Prefer clear module boundaries over large mixed-purpose files.
- Add tests for data transforms, coordinate conversions, output decoding, and metrics.
- Avoid speculative abstractions before the training and inference path is concrete.
- Preserve Rust idioms and use explicit types where they improve model/data correctness.
- Keep the project clean: do not leave unused, speculative, optional, fallback, or "maybe later" dependencies, features, modules, files, configs, scripts, generated artifacts, or compatibility shims in the repository.
- Do not lower, remove, hide, or weaken `DESIGN.md` requirements merely to make the current checkout compile. If a required dependency, backend, feature, or implementation path cannot be made to work in the current environment, keep the requirement visible and state the exact blocker.
- Do not keep broken native-performance dependencies as dormant project baggage. If a native high-performance dependency is required by the design but cannot compile, report the environment/toolchain issue instead of replacing it with a lower-performance path without explicit user approval.
- Prefer the fastest dependency that is compatible with the project requirements. Do not add slower fallbacks, optional alternatives, or compatibility shims unless the user explicitly requests that tradeoff.

## Commit Discipline

- After completing any task in this project, create a Git commit for the completed work.
