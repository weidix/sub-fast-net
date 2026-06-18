# Model

SubFastNet is a subtitle detector, not OCR.

The default model is a compact TextNet-style / FAST-like detector:

```text
TextNetStem: stride-2 reduction plus asymmetric 3x1/1x3 text blocks
TextNetStage P2: stride 4 feature for low-height subtitle localization
TextNetStage P3: stride 8 feature for the main text-detection signal
TextNetStage P4: stride 16 semantic feature
1x1 alignment convolutions to a small fusion width
nearest upsample to P2 resolution
additive lightweight feature fusion
separate FAST-like region and kernel heads
```

The code-level hierarchy mirrors this:

```text
SubFastNet
  TextNetStem: stride 2
  TextNetStage stride4 -> P2
  TextNetStage stride8 -> P3
  TextNetStage stride16 -> P4
  align4 / align8 / align16: 1x1 channel alignment
  nearest upsample P3/P4 to P2
  TextNetBlock fusion_refine
  DetectionHead region_head
  DetectionHead kernel_head
```

`architecture_spec()` exposes the contract used by tests: P2/P3/P4 strides are 4/8/16, the output stride is 4, and the default detector has exactly two heads: region and kernel. This stays aligned with FAST/TextNet's multi-scale text feature idea without adding a heavy FPN/PAN or an OCR branch.

Outputs:

```text
text_region_logits: [B, 1, H, W]
kernel_logits: [B, 1, H, W]
```

Variants:

```text
tiny: default high-FPS model
small: speed/accuracy balance
base: larger experiment
```

The model does not include MobileNet, YOLO, ResNet, EfficientNet, recognition branches, character classifiers, CTC, or attention decoding.

Model size reporting has two fields:

```text
model_size_bytes_estimate: parameter-count estimate
final_model_artifact_size_bytes: actual saved checkpoint model artifact size
```

The actual artifact size is only available after training saves `final/model.bin`.
