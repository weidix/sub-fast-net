# Inference

Inference reads an image, applies deterministic aligned short resize, runs SubFastNet, postprocesses region/kernel logits, and restores boxes to original pixel coordinates.

Output format:

```json
{
  "image": "sample.jpg",
  "width": 1920,
  "height": 1080,
  "boxes": [
    {"x1": 100.0, "y1": 720.0, "x2": 820.0, "y2": 770.0, "confidence": 0.94}
  ],
  "meta": {"source": null, "frame_id": null}
}
```

Postprocessing applies sigmoid, thresholds region and kernel maps, extracts kernel connected components as seeds, expands each seed inside the thresholded region mask with lightweight BFS aggregation, computes bbox confidence, filters small boxes, and restores coordinates using recorded scale and padding.
