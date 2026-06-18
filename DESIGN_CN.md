# SubFastNet 项目生成需求文档

## 1. 项目定位

你是一个资深 Rust / Burn 深度学习工程 Agent。请生成一个完整、工程级、可训练、可验证、可推理、可 benchmark 的 Rust Burn 项目，用于训练一个专注于字幕区域检测的高 FPS 模型。

项目名称：SubFastNet

任务类型：字幕区域检测

这不是 OCR 项目，不做字符识别，不做字幕文本内容识别，不做端到端 OCR，不做 CTC，不做 Attention Decoder，不做文本解码。

输入是图片。

输出是字幕区域的像素坐标 bbox 和 confidence。

默认推理输出格式：

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

项目核心目标是训练一个参考 FAST / TextNet 思路、但专门针对字幕检测场景简化和优化的模型。重点目标是桌面端高 FPS 推理，同时支持 CPU、CUDA、WGPU 训练与推理。

## 2. 已确认事实

以下事项已经明确，不要再向用户确认：

1. 当前任务是字幕区域检测，不是 OCR。
2. 不需要字符识别。
3. 不需要 CER、WER、文本 exact match 等 OCR 指标。
4. 所有 bbox 都是字幕框。
5. 这是单类检测模型。
6. `labels/*.txt` 是 YOLO bbox 格式。
7. 双行字幕的标注方式是：每一行一个 bbox。
8. 模型按训练样本中的 bbox 学习，不需要跨行合并。
9. 训练集由多个 root 组成。
10. 验证集只有一个 root。
11. 每个 root 内部结构一致。
12. `label_masks.json` 必须参与训练样本注入。
13. `annotations.jsonl` 必须读取并作为元信息参与数据校验、错误分析和输出。
14. 推理输出使用像素坐标 bbox。
15. 部署目标主要是桌面端，高优先级目标是推理 FPS 更快。
16. 模型名称必须是 `SubFastNet`。
17. 模型默认架构必须参考 FAST 使用的 `TextNet`。
18. 不要默认使用 MobileNet、LCNet、ShuffleNet、YOLO backbone 或其他通用分类 backbone。

## 3. 数据集结构

训练集支持多个 root。

验证集支持一个 root。

每个 root 的结构固定如下：

```text
dataset_root/
├── images/
│   └── *.jpg
├── labels/
│   └── *.txt
├── annotations.jsonl
└── label_masks.json
```

`images/` 存放 jpg 图片。

`labels/` 存放与图片同名的 txt 标签文件。

`annotations.jsonl` 存放每张图片的来源、尺寸、帧号、检测框等元信息。

`label_masks.json` 存放人工屏蔽、删除、修正或补充标注的记录。

训练配置中必须支持多个训练 root，例如：

```toml
train_roots = [
  "/data/subtitle/train_a",
  "/data/subtitle/train_b",
  "/data/subtitle/train_c"
]

val_root = "/data/subtitle/val"
```

## 4. Label 格式

`labels/*.txt` 使用 YOLO bbox 格式。

每一行表示一个字幕框：

```text
class_id x_center y_center width height
```

示例：

```text
0 0.404167 0.962963 0.158333 0.066667
0 0.624479 0.961574 0.098958 0.065741
```

字段含义：

* `class_id`：类别 ID，固定视为字幕类别。
* `x_center`：bbox 中心点 x，归一化坐标。
* `y_center`：bbox 中心点 y，归一化坐标。
* `width`：bbox 宽度，归一化坐标。
* `height`：bbox 高度，归一化坐标。

所有 bbox 都是字幕区域。

这是单类模型，不需要多类别分类头。

解析规则：

1. 每行必须有 5 列。
2. `class_id` 可解析为整数。
3. `x_center`、`y_center`、`width`、`height` 可解析为浮点数。
4. `width > 0`。
5. `height > 0`。
6. bbox 应位于合理范围内。
7. 越界 bbox 可根据 strict 策略报错或裁剪。
8. 空 label 文件表示该图无字幕样本，除非配置要求 strict 报错。
9. label 缺失时按 strict 策略处理。

## 5. label_masks.json 处理要求

`label_masks.json` 必须参与训练样本注入流程，不能忽略。

加载样本时必须按以下顺序处理：

1. 读取图片。
2. 读取同名 `labels/*.txt`。
3. 读取 `annotations.jsonl` 中对应图片的元信息。
4. 读取 `label_masks.json` 中对应图片的人工屏蔽或修正规则。
5. 先应用 `label_masks.json`。
6. 再生成最终训练 target。

`label_masks.json` 可能包含以下行为：

* 忽略整张图片。
* 删除某个 bbox。
* 修正某个 bbox。
* 增加人工补充 bbox。
* 增加 ignore region。
* 标记某些区域不参与 loss。
* 标记某些原始 label 不可信。

Agent 必须读取真实文件并根据 schema 实现适配。只有当 `label_masks.json` 的 schema 无法从文件内容推断、存在冲突、或会改变训练标签语义时，才允许向用户确认。

推荐设计一个兼容层：

```text
RawLabelMaskRecord
LabelMaskAction
ApplyLabelMaskResult
```

推荐处理优先级：

1. drop image
2. remove bbox
3. correct bbox
4. add bbox
5. add ignore region
6. final validation

`label_masks.json` 应用于原始 label 之后、target mask 生成之前。

最终进入训练的 bbox 必须是经过 `label_masks.json` 处理后的 bbox。

ignore region 必须进入 `training_mask`，并在 loss 计算时排除。

## 6. annotations.jsonl 处理要求

`annotations.jsonl` 必须读取。

默认情况下，训练 label 以 `labels/*.txt` 为准。

`annotations.jsonl` 不直接覆盖 label，除非真实数据中存在明确覆盖规则。

`annotations.jsonl` 用途：

1. 校验图片尺寸。
2. 记录 source。
3. 记录 frame_id。
4. 记录视频来源。
5. 记录原始检测框元信息。
6. 用于错误样本分析。
7. 用于推理输出补充 meta。
8. 用于数据集检查报告。
9. 用于 debug 和统计分析。

如果 `annotations.jsonl` schema 能从文件内容推断，Agent 自行实现适配。

如果 schema 无法推断，或其中字段会改变 label 语义，再向用户确认。

## 7. Dataset 设计

必须实现并兼容 Burn 当前 Dataset 抽象。

Dataset 必须支持：

1. 多训练 root。
2. 单验证 root。
3. 每个 root 独立读取 images、labels、annotations、label_masks。
4. root 间样本合并。
5. 样本唯一 ID。
6. 异常样本处理。
7. strict / non-strict 模式。
8. 数据集检查命令。

Dataset item 至少包含：

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
image_data 或 image_tensor
gt_text
gt_kernel
training_mask
gt_instance
gt_boxes
img_meta
```

strict 模式：

```toml
strict_dataset = true
```

行为：

* 图片缺失：报错。
* label 缺失：报错。
* label 格式错误：报错。
* bbox 非法：报错。
* annotations 关键字段冲突：报错。
* label_masks schema 冲突：报错。

non-strict 模式：

```toml
strict_dataset = false
```

行为：

* 异常样本跳过或修复。
* 输出 warning。
* 在数据集检查报告中记录。
* 训练不中断。

必须实现 `inspect-dataset` 命令，输出：

1. 样本总数。
2. 每个 root 的样本数。
3. 空 label 数。
4. 无效 label 数。
5. 被 label_masks 删除的 bbox 数。
6. 被 label_masks 修正的 bbox 数。
7. ignore region 数。
8. 图片尺寸分布。
9. bbox 尺寸分布。
10. 字幕位置分布。
11. 异常样本列表。

## 8. 预处理设计

SubFastNet 的预处理必须参考 FAST 的训练数据处理流程，而不是使用普通目标检测项目的简化 resize + bbox pipeline。

FAST 的预处理核心不是单纯的图像缩放，而是围绕以下训练目标构建：

```text
gt_instance
gt_text
gt_kernel
training_mask
```

SubFastNet 必须继承这个思路，并针对字幕 bbox 数据做适配。

由于本项目的 label 是 YOLO bbox，而不是 FAST 原始数据集中的 polygon，因此 SubFastNet 的预处理规则是：

```text
YOLO bbox -> pixel bbox -> rectangle polygon -> gt_instance / gt_text / gt_kernel / training_mask
```

也就是说，虽然原始标注是 bbox，但在 target 生成阶段应把 bbox 视为四点矩形 polygon，以便沿用 FAST 的 mask-based target 生成流程。

### 8.1 训练预处理主流程

训练样本处理顺序必须为：

1. 读取图片。
2. 解码图片。
3. 转换为 RGB。
4. 读取同名 `labels/*.txt`。
5. 解析 YOLO bbox。
6. 读取 `annotations.jsonl` 中对应图片的元信息。
7. 读取并应用 `label_masks.json`。
8. 将 YOLO normalized bbox 转为原图像素 bbox。
9. 将 bbox 转为四点矩形 polygon。
10. 执行 FAST-style random scale。
11. 同步缩放 image、bbox、polygon。
12. 初始化 `gt_instance`。
13. 初始化 `training_mask`。
14. 将每个有效字幕 bbox 绘制到 `gt_instance`。
15. 将 ignore region 绘制到 `training_mask = 0`。
16. 为每个实例生成 instance-level kernel source mask。
17. 使用 FAST-style min pooling / pooling kernel 生成合并后的 `gt_kernel`。
18. 使用 shrink 后的 bbox / polygon 生成补充 kernel。
19. 将 min-pooled kernel 与 shrink kernel 合并。
20. 执行 FAST-style random horizontal flip。
21. 执行 FAST-style random rotate。
22. 执行 FAST-style random crop padding。
23. 从 `gt_instance` 生成 `gt_text`。
24. 执行 color jitter / blur 等图像增强。
25. 转 tensor。
26. normalize。
27. batch collate。

最终训练 batch 至少包含：

```text
imgs
gt_texts
gt_kernels
training_masks
gt_instances
gt_boxes
img_metas
```

其中：

```text
gt_texts       = 二值字幕区域 mask
gt_kernels     = FAST-style kernel mask
training_masks = loss ignore mask
gt_instances   = instance id mask
gt_boxes       = 经过 label_masks 和几何变换后的 bbox
img_metas      = 原图尺寸、缩放比例、padding、source、frame_id 等元信息
```

### 8.2 验证预处理主流程

验证阶段必须使用 deterministic preprocess。

验证样本处理顺序：

1. 读取图片。
2. 解码图片。
3. 转换为 RGB。
4. 读取 YOLO bbox。
5. 应用 `label_masks.json`。
6. YOLO bbox 转像素 bbox。
7. bbox 转矩形 polygon。
8. 使用 FAST-style aligned short resize。
9. 记录原图尺寸。
10. 记录 resize 后尺寸。
11. 同步变换 bbox。
12. 生成 `gt_text`、`gt_kernel`、`training_mask`、`gt_instance`。
13. 转 tensor。
14. normalize。
15. batch collate。

验证阶段禁止随机增强。

验证阶段必须保存：

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

这些信息用于将预测 bbox 恢复到原图像素坐标。

### 8.3 推理预处理主流程

推理阶段不读取 label。

推理样本处理顺序：

1. 读取图片。
2. 解码图片。
3. 转换为 RGB。
4. 使用 FAST-style aligned short resize。
5. 记录原图尺寸。
6. 记录 resize 后尺寸。
7. 转 tensor。
8. normalize。
9. 输入模型。
10. 后处理输出 bbox。
11. bbox 恢复到原图像素坐标。

推理阶段不得使用随机增强。

### 8.4 FAST-style aligned resize

SubFastNet 必须实现 FAST-style `scale_aligned_short`。

行为：

1. 按 `short_size` 缩放图片短边。
2. 保持宽高比。
3. 将缩放后的宽高对齐到模型 stride 或配置的 alignment。
4. 记录缩放比例。
5. 后处理时用记录的比例恢复到原图坐标。

默认：

```toml
short_size = 640
alignment = 32
```

对于字幕检测，可以允许使用非正方形输入，但必须保持 FAST-style aligned resize 的语义。

如果为了训练 batch 固定尺寸而使用 crop / padding，应只在训练阶段使用。

验证和推理阶段以 aligned short resize 为默认策略。

### 8.5 FAST-style random scale

训练阶段必须实现 FAST-style random scale。

默认策略：

```toml
scale_min = 0.7
scale_max = 1.3
aspect_min = 0.9
aspect_max = 1.1
```

对于字幕数据，也可以保留 FAST Total-Text 风格的较大 scale range 作为可选实验：

```toml
scale_min = 0.5
scale_max = 2.0
```

但默认建议先使用较稳的：

```toml
scale_min = 0.7
scale_max = 1.3
aspect_min = 0.9
aspect_max = 1.1
```

random scale 必须同步变换：

```text
image
bbox
rectangle polygon
ignore region
```

### 8.6 FAST-style random crop padding

训练阶段必须实现 FAST-style `random_crop_padding`。

它不是普通 random crop。

要求：

1. crop 到固定训练尺寸。
2. 如果图片小于目标尺寸，则 padding。
3. crop 必须同步作用于 image、gt_instance、training_mask、gt_kernel。
4. crop 后仍需保留有效字幕区域。
5. crop 后 bbox 要重新裁剪并校验。
6. 完全落在 crop 外的 bbox 删除。
7. 部分落在 crop 内的 bbox 裁剪到 crop 边界。
8. 裁剪后过小 bbox 删除或进入 ignore，具体由 strict 策略控制。

默认：

```toml
input_size = 640
```

如果使用非正方形字幕训练尺寸，例如 640x384，也必须保持同样语义：

```toml
input_width = 640
input_height = 384
```

### 8.7 FAST-style random rotate

训练阶段必须实现 FAST-style random rotate，但字幕场景默认角度应更小。

FAST 在通用文本检测中会使用较大的旋转增强；SubFastNet 面向字幕检测，字幕通常水平或近水平，因此默认建议：

```toml
random_rotate = true
rotate_angle = 5
```

最大不建议超过：

```toml
rotate_angle = 10
```

只有当训练数据中确实存在明显倾斜字幕时，才允许更大的角度。

旋转必须同步作用于：

```text
image
gt_instance
training_mask
gt_kernel
bbox
rectangle polygon
ignore region
```

旋转后 bbox 应重新由 polygon 的外接矩形生成。

### 8.8 FAST-style random horizontal flip

训练阶段可以实现 FAST-style random horizontal flip。

字幕检测只关心区域，不识别文字内容，因此 horizontal flip 不会破坏 OCR 标签，因为本项目没有 OCR 标签。

如果后续引入 OCR，horizontal flip 必须关闭。

默认：

```toml
random_horizontal_flip = true
flip_prob = 0.5
```

### 8.9 ColorJitter 与 Blur

训练阶段保留 FAST-style 图像增强。

默认增强：

```text
brightness jitter
contrast jitter
saturation jitter
optional gaussian blur
```

建议默认：

```toml
brightness = 0.125
contrast = 0.4
saturation = 0.4
hue = 0.1
gaussian_blur = true
gaussian_blur_prob = 0.5
```

由于字幕可能有描边、阴影、压缩噪声，允许增加轻量 jpeg compression noise，但它应作为字幕场景扩展增强，不应替代 FAST 原始增强。

### 8.10 Normalize

SubFastNet 默认使用 ImageNet mean/std，与 FAST 训练习惯保持一致：

```toml
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
```

如果从零训练且没有 ImageNet / TextNet 预训练，也可以通过配置改为 dataset mean/std，但默认文档和实现应先对齐 FAST。

### 8.11 预处理算子列表

必须实现以下算子：

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

每个算子必须说明：

1. 输入。
2. 输出。
3. 参数。
4. 是否训练阶段使用。
5. 是否验证阶段使用。
6. 是否推理阶段使用。
7. 是否同步变换 bbox / polygon / mask。
8. strict / non-strict 下的错误处理。

## 9. FAST-style Target 生成

SubFastNet 的 target 生成必须以 FAST 为主线。

原始 FAST 的训练 target 包括：

```text
gt_texts
gt_kernels
training_masks
gt_instances
```

SubFastNet 必须保留这些核心 target。

### 9.1 从 YOLO bbox 到 instance mask

由于数据集标签是 YOLO bbox，每个 bbox 都是字幕框。

处理步骤：

1. 读取 YOLO normalized bbox。
2. 转为原图像素 bbox。
3. 应用 `label_masks.json`。
4. 将 bbox 转成四点矩形 polygon：

```text
(x1, y1)
(x2, y1)
(x2, y2)
(x1, y2)
```

5. 对每个有效 bbox 分配 instance id：

```text
instance_id = index + 1
```

6. 将该 rectangle polygon 绘制到 `gt_instance`。

`gt_instance` 语义：

```text
0 = background
1..N = subtitle instance id
```

### 9.2 training_mask

`training_mask` 默认全 1。

以下区域置 0：

1. `label_masks.json` 指定 ignore 的区域。
2. 被人工标记为不可信的 bbox 区域。
3. 非法 bbox 区域。
4. 太小、无法稳定训练的 bbox 区域。
5. strict=false 时被跳过但仍需屏蔽的区域。

`training_mask = 0` 的区域不参与 loss。

### 9.3 gt_text

`gt_text` 从 `gt_instance` 生成：

```text
gt_text = gt_instance > 0
```

语义：

```text
1 = subtitle text region
0 = background
```

### 9.4 FAST-style kernel mask

SubFastNet 的 `gt_kernel` 必须参考 FAST 的 kernel 生成方式，不应只做简单 bbox shrink。

推荐流程：

1. 为每个 instance 生成单独的 instance kernel source mask。
2. 使用 min pooling 思路压缩每个 instance 内部区域。
3. 检测 overlap region。
4. overlap region 从 kernel 中移除。
5. 对 bbox / polygon 做 shrink。
6. 生成 shrink kernel。
7. 将 min-pooled kernel 与 shrink kernel 合并。
8. 输出最终二值 `gt_kernel`。

目标：

```text
gt_kernel = stable center area of each subtitle instance
```

kernel 的作用：

1. 作为后处理 seed。
2. 区分相邻字幕行。
3. 区分相邻字幕框。
4. 降低复杂后处理成本。

### 9.5 pooling_size

必须支持 FAST-style pooling size。

默认：

```toml
pooling_size = 9
```

含义：

1. 控制 min pooling kernel 大小。
2. 控制 kernel 收缩强度。
3. 影响相邻实例分离能力。
4. 影响小字幕框保留能力。

对于字幕框，必须加入保护：

```toml
min_kernel_width = 3
min_kernel_height = 3
```

防止小高度字幕被 kernel 操作抹掉。

### 9.6 shrink_kernel_scale

必须支持 FAST-style shrink kernel scale。

默认：

```toml
shrink_kernel_scale = 0.1
```

注意：这里不是普通检测中常见的 bbox shrink ratio 0.5 / 0.7。FAST 中 shrink scale 更接近“边界内缩比例”，它与 min pooling 共同构成最终 kernel。

SubFastNet 应先使用 FAST-like 默认值，再通过 benchmark / validation 调整。

推荐初始配置：

```toml
pooling_size = 9
shrink_kernel_scale = 0.1
```

### 9.7 双行字幕处理

双行字幕是一行一个 bbox。

因此每一行分别作为一个 instance：

```text
line_1 -> instance 1
line_2 -> instance 2
```

不要合并双行字幕。

不要把两行合成一个 bbox。

kernel 生成时必须确保上下两行字幕能被分开。

### 9.8 Target 输出

训练样本最终输出：

```text
imgs: [3, H, W]
gt_texts: [H, W]
gt_kernels: [H, W]
training_masks: [H, W]
gt_instances: [H, W]
gt_boxes: Vec<Box>
img_metas: ImageMeta
```

batch 后输出：

```text
imgs: [B, 3, H, W]
gt_texts: [B, H, W]
gt_kernels: [B, H, W]
training_masks: [B, H, W]
gt_instances: [B, H, W]
gt_boxes: Vec<Vec<Box>>
img_metas: Vec<ImageMeta>
```

## 10. SubFastNet 训练预处理默认配置

默认配置应以 FAST 风格为主：

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

如果为了字幕场景使用非正方形输入，应仍保留 FAST-style crop padding 和 aligned resize 语义：

```toml
input_width = 640
input_height = 384
short_size = 384
alignment = 32
```

不要把预处理退化为普通 YOLO 检测 pipeline。

不要只生成 bbox target。

不要跳过 `gt_instance`。

不要跳过 `gt_kernel`。

不要跳过 `training_mask`。

不要忽略 `label_masks.json`。

## 11. 与 FAST 的差异边界

SubFastNet 允许和 FAST 不同的地方：

1. 原始 label 来源是 YOLO bbox，而不是 polygon。
2. bbox 会转换成 rectangle polygon 后再进入 FAST-style target pipeline。
3. 输出是 axis-aligned 像素 bbox，不输出 polygon。
4. 字幕场景默认使用较小旋转角度。
5. 字幕场景默认不做任意形状文本增强。
6. 字幕场景不做 OCR。
7. 字幕场景不做跨语种文本识别。
8. 后处理只需要得到 bbox，不需要恢复复杂 text polygon。

SubFastNet 不应该和 FAST 不同的地方：

1. 不应该取消 `gt_instance`。
2. 不应该取消 `gt_text`。
3. 不应该取消 `gt_kernel`。
4. 不应该取消 `training_mask`。
5. 不应该把 kernel 生成简化成普通 bbox shrink 后就结束。
6. 不应该把训练预处理改成普通 YOLO 检测增强。
7. 不应该忽略 FAST-style scale / crop padding / aligned resize。
8. 不应该忽略 pooling_size。
9. 不应该忽略 shrink_kernel_scale。
10. 不应该忽略 min pooling / overlap suppression 的 kernel 思路。

## 12. 模型架构：SubFastNet

模型名称必须是：

```text
SubFastNet
```

默认架构必须参考 FAST 的 TextNet。

不要默认使用：

* MobileNet
* MobileNetV2
* MobileNetV3
* LCNet
* ShuffleNet
* YOLO backbone
* EfficientNet
* ResNet

这些最多只能作为 baseline 或 ablation，不是主线架构。

主线架构：

```text
SubFastNet = TextNet-style backbone + lightweight feature fusion + FAST-like minimalist kernel detection head
```

TextNet 是 FAST 中面向文本检测任务使用的 backbone。Agent 应优先参考 FAST 原论文和官方实现中的 TextNet-T / TextNet-S / TextNet-B 结构。如果项目本地已有 FAST / TextNet 实现或说明，必须优先读取并对齐。

模型 variants：

```text
SubFastNet-Tiny
SubFastNet-Small
SubFastNet-Base
```

默认首选：

```text
SubFastNet-Tiny
```

默认用途：

* Tiny：最高 FPS。
* Small：速度和精度平衡。
* Base：可选高精度实验，不作为默认。

TextNet-style backbone 要求：

1. 面向文本检测，而不是分类迁移优先。
2. 输出多尺度特征。
3. 保持对细长字幕区域的定位能力。
4. 支持 stride 4 / stride 8 / stride 16 特征。
5. stride 32 特征只在 Small / Base 中可选。
6. 优先使用 Burn CPU / CUDA / WGPU 稳定支持的算子。
7. 不依赖自定义 CUDA kernel。
8. 不依赖复杂动态 shape。
9. 不引入 OCR 分支。

推荐特征层：

```text
P2: stride 4，可选，用于小高度字幕
P3: stride 8，主检测特征
P4: stride 16，语义增强
P5: stride 32，仅 Small/Base 可选
```

轻量特征融合：

1. 1x1 conv 对齐通道。
2. nearest upsample。
3. add 或 concat 融合。
4. lightweight conv refine。
5. 输出统一检测特征图。

不要使用重型 FPN / PAN。

不要为了泛化自然场景文本而引入复杂 polygon 分支。

## 13. Detection Head

Detection head 使用 FAST-like minimalist kernel representation。

默认输出两个 logits：

```text
text_region_logits: [B, 1, H, W]
kernel_logits: [B, 1, H, W]
```

region head：

* 预测完整字幕区域。
* 监督信号来自 `gt_text`。

kernel head：

* 预测 shrink / pooling 后的字幕 kernel。
* 监督信号来自 `gt_kernel`。
* 用于后处理阶段区分相邻字幕框。

默认不启用 bbox regression head。

如果 region / kernel 后处理得到的 bbox 精度不足，可以增加轻量 bbox refinement head，但这不是默认主线。

禁止默认添加 OCR head。

禁止默认添加 text recognition head。

禁止默认添加 character classification head。

## 14. Loss 设计

默认 loss：

```text
total_loss = region_loss * region_weight
           + kernel_loss * kernel_weight
           + optional_bbox_loss * bbox_weight
```

region loss：

```text
BCEWithLogits + Dice Loss
```

kernel loss：

```text
BCEWithLogits + Dice Loss
```

bbox loss：

默认关闭。

如果启用 bbox refinement head，可使用：

```text
SmoothL1
IoU loss
GIoU loss
```

ignore mask：

1. region loss 必须支持 `training_mask`。
2. kernel loss 必须支持 `training_mask`。
3. ignore 区域不参与正负样本统计。
4. label_masks.json 产生的 ignore region 必须生效。

可选 OHEM：

* 用于控制背景区域过多的问题。
* 默认可关闭。
* 如果启用，应仅作用于 region loss 的负样本区域。

必须输出 loss breakdown：

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

不要实现：

```text
CTC loss
attention decoder loss
CER
WER
text exact match
```

## 15. 后处理设计

后处理必须简单、高速、适合桌面端高 FPS。

默认流程：

1. region logits sigmoid。
2. kernel logits sigmoid。
3. region threshold。
4. kernel threshold。
5. kernel 作为 seed。
6. 在 region 内做轻量聚合。
7. connected component 或等价快速聚合。
8. 每个 component 生成 axis-aligned bbox。
9. 计算 confidence。
10. 过滤过小 bbox。
11. optional NMS。
12. 恢复到原图像素坐标。
13. 输出 `x1, y1, x2, y2, confidence`。

输出 bbox 必须是像素坐标。

默认不输出 polygon。

默认不做复杂曲线文本恢复。

默认不做跨帧字幕合并。

bbox restore 必须考虑：

1. resize scale。
2. letterbox padding。
3. 原图宽高。
4. 坐标裁剪。
5. 最小 bbox 尺寸过滤。

confidence 建议：

```text
confidence = mean(region_prob within component)
```

或：

```text
confidence = weighted_mean(region_prob, kernel_prob)
```

必须 benchmark 后处理耗时。

后处理应输出：

```text
postprocess_latency_ms
candidate_count
final_box_count
```

## 16. 训练流程

训练流程必须包括：

1. 读取配置文件。
2. 初始化 backend：CPU / CUDA / WGPU。
3. 设置 seed。
4. 加载多个 train root。
5. 加载单个 val root。
6. 构建 Burn Dataset。
7. 构建 dataloader。
8. 构建 SubFastNet。
9. 初始化 optimizer。
10. 初始化 scheduler。
11. 接入 Burn learner 或标准 Burn training loop。
12. 接入 Burn TUI。
13. 按 interval 验证。
14. 保存 checkpoint。
15. 支持 resume。
16. 保存 best model。
17. 保存 final model。
18. 输出 metrics。
19. 输出 training summary。
20. 输出错误样本分析。

训练阶段必须输出：

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
gpu_memory 或 backend memory 信息
```

## 17. 验证流程

验证集只有一个 root。

验证阶段不得使用随机增强。

验证流程：

1. 加载 val root。
2. 应用与训练一致的 deterministic preprocess。
3. batch 推理。
4. 计算验证 loss。
5. 执行后处理。
6. 将预测 bbox 恢复到原图像素坐标。
7. 将 YOLO label 转换为真实像素 bbox。
8. 应用 label_masks.json 后再作为 GT。
9. 计算 bbox IoU。
10. 按 IoU threshold 匹配 TP / FP / FN。
11. 计算 precision。
12. 计算 recall。
13. 计算 F1。
14. 计算 mean IoU。
15. 计算 FPS。
16. 计算 latency p50 / p95。
17. 保存错误样本分析。

验证指标：

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

错误样本分析输出：

```text
false_positive.jsonl
false_negative.jsonl
low_iou.jsonl
ignored_samples.jsonl
```

每条错误记录建议包含：

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

## 18. Metrics 设计

训练 metrics：

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

验证 metrics：

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

benchmark metrics：

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

所有 metrics 至少输出到：

```text
console
Burn TUI
metrics.jsonl
summary.json
```

## 19. Burn TUI 要求

必须接入标准化 Burn TUI。

TUI 至少显示：

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

如果 backend 可提供 memory 信息，也显示：

```text
memory usage
gpu memory
```

TUI 不应破坏普通日志输出。

必须支持关闭：

```toml
tui_enabled = false
```

## 20. Backend 支持

必须支持：

```text
cpu
cuda
wgpu
```

配置示例：

```toml
backend = "wgpu"
```

要求：

1. CPU 可训练。
2. CUDA 可训练。
3. WGPU 可训练。
4. 同一套 Dataset / preprocess / model / loss / metrics 尽量复用。
5. backend 差异只放在初始化和 feature gate 中。
6. 不要依赖只在 CUDA 可用的自定义 kernel。
7. 后处理默认可以在 CPU 完成，但必须测量耗时。
8. 如果实现 GPU 后处理，必须保留 CPU fallback。

## 21. 配置文件要求

配置文件保持必要且简洁。

不要设计过度复杂的 schema。

不要把训练参数硬编码。

最低配置字段：

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

允许 Agent 根据实现需要自然补充少量字段，但不要生成几十个不必要配置项。

## 22. CLI 要求

必须提供命令行接口。

命令至少包括：

```text
train
validate
infer
inspect-dataset
benchmark
```

示例：

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

infer 命令输出像素坐标 bbox。

## 23. Benchmark 要求

必须实现 benchmark。

benchmark 内容：

1. dataloader throughput。
2. preprocess throughput。
3. train step time。
4. validation step time。
5. inference FPS。
6. latency p50。
7. latency p95。
8. postprocess latency。
9. memory usage。
10. end-to-end latency。

benchmark 输出：

```text
console
Burn TUI
metrics.jsonl
summary.json
```

benchmark 必须区分：

```text
preprocess time
model forward time
postprocess time
end-to-end time
```

FPS 目标以端到端推理为准，不只看模型 forward。

## 24. Checkpoint 和 Resume

必须支持：

1. 定期保存 checkpoint。
2. 保存 best model。
3. 保存 final model。
4. 从 checkpoint resume。
5. 保存 optimizer state。
6. 保存 scheduler state。
7. 保存 epoch / step。
8. 保存 config snapshot。
9. 保存 metrics summary。

输出目录建议：

```text
outputs/
└── subfastnet_tiny/
    ├── checkpoints/
    ├── best/
    ├── final/
    ├── metrics.jsonl
    ├── summary.json
    ├── config.snapshot.toml
    └── errors/
```

## 25. 文档要求

必须生成详细文档，但不要过度展开每个 `.rs` 文件的内部细节。

至少生成根目录文档：

```text
README.md
DESIGN.md
```

主题文档保留在 `docs/` 目录下：

```text
docs/DATASET.md
docs/PREPROCESSING.md
docs/MODEL.md
docs/TRAINING.md
docs/METRICS.md
docs/INFERENCE.md
docs/BENCHMARK.md
```

文档必须说明：

1. SubFastNet 是字幕区域检测模型。
2. SubFastNet 不是 OCR。
3. 数据集结构。
4. YOLO label 解析。
5. 多训练 root 加载方式。
6. 单验证 root 加载方式。
7. `label_masks.json` 如何参与训练样本注入。
8. `annotations.jsonl` 如何作为元信息使用。
9. FAST-style 预处理流程。
10. `gt_instance` 生成。
11. `gt_text` 生成。
12. `gt_kernel` 生成。
13. `training_mask` 生成。
14. TextNet-style backbone。
15. FAST-like minimalist kernel head。
16. loss 设计。
17. bbox metric 计算。
18. 高 FPS 推理优化点。
19. CPU / CUDA / WGPU backend 注意事项。
20. Burn TUI。
21. checkpoint / resume。
22. benchmark 方法。
23. 错误样本分析。

## 26. 测试要求

必须包含测试。

测试至少覆盖：

1. YOLO label parser。
2. bbox normalized -> pixel 转换。
3. bbox -> rectangle polygon 转换。
4. resize / aligned short resize bbox 同步变换。
5. random scale bbox / polygon 同步变换。
6. random rotate bbox / polygon 同步变换。
7. random crop padding mask 同步变换。
8. label_masks 应用。
9. annotations.jsonl 读取。
10. gt_instance 生成。
11. gt_text 生成。
12. gt_kernel 生成。
13. training_mask 生成。
14. Dataset 多 root 合并。
15. empty label 处理。
16. strict / non-strict 行为。
17. postprocess bbox restore。
18. metric IoU matching。
19. smoke train。

smoke train 应能用极小数据集跑通：

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

## 27. 工程模块职责

可以使用标准 Rust 单 crate 或 workspace。

不要在设计文档中过度列出每个 `.rs` 文件细节，但必须包含以下模块职责：

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

职责说明：

config：

* 读取 TOML 配置。
* 校验必填字段。
* 保存 config snapshot。

dataset：

* 加载多个 train root。
* 加载单个 val root。
* 解析 images / labels / annotations / label_masks。
* 实现 Burn Dataset 兼容结构。

preprocess：

* 图片解码。
* FAST-style random scale。
* FAST-style random crop padding。
* FAST-style random rotate。
* FAST-style horizontal flip。
* aligned short resize。
* normalize。
* bbox / polygon / mask 同步变换。
* tensor 转换。

target：

* 生成 gt_instance。
* 生成 gt_text。
* 生成 gt_kernel。
* 生成 training_mask。
* 生成 gt_boxes。

model：

* 实现 SubFastNet。
* 实现 TextNet-style backbone。
* 实现轻量特征融合。
* 实现 FAST-like region/kernel head。

loss：

* region loss。
* kernel loss。
* optional bbox loss。
* training_mask ignore 支持。

metrics：

* bbox IoU。
* precision / recall / F1。
* mean IoU。
* latency / FPS。

train：

* 训练 loop。
* optimizer。
* scheduler。
* Burn TUI。
* checkpoint。

validate：

* 验证 loop。
* 错误样本分析。

infer：

* 单图推理。
* 批量推理。
* 输出像素坐标 bbox。

postprocess：

* sigmoid。
* threshold。
* component 聚合。
* bbox restore。
* NMS 或轻量过滤。

benchmark：

* preprocess。
* forward。
* postprocess。
* end-to-end FPS。

## 28. 禁止事项

不要默认使用 Python。

不要默认使用 PyTorch。

不要把任务做成 OCR。

不要实现字符识别头。

不要实现 CTC。

不要实现 attention decoder。

不要输出 CER / WER。

不要再询问任务是不是 OCR。

不要再询问 bbox 是否是字幕框。

不要再询问是否单类。

不要再询问双行字幕如何标注。

不要把训练集假设为单目录。

不要把验证集假设为多个目录。

不要忽略 `label_masks.json`。

不要忽略 `annotations.jsonl`。

不要过度设计配置文件。

不要把训练参数硬编码。

不要默认使用 MobileNetV3。

不要默认使用 LCNet。

不要默认使用 ShuffleNet。

不要默认使用 YOLO backbone。

不要默认支持任意弯曲文本或复杂 polygon。

不要使用复杂后处理牺牲 FPS。

不要默认做跨帧字幕合并。

不要把 FAST-style 预处理退化成普通 YOLO 检测预处理。

不要跳过 `gt_instance`。

不要跳过 `gt_text`。

不要跳过 `gt_kernel`。

不要跳过 `training_mask`。

不要把文档写成只有概念、没有可落地工程设计。

## 29. 允许确认的问题

只有以下情况允许向用户确认：

1. `label_masks.json` 的真实 schema 无法从文件推断。
2. `label_masks.json` 中存在互相冲突的规则。
3. `annotations.jsonl` 中字段会改变 label 语义。
4. FAST / TextNet 的具体结构在论文、官方实现、项目文件中都无法确定。
5. 某个实现决策会直接改变训练 target。
6. 某个实现决策会直接改变 loss。
7. 某个实现决策会直接改变推理输出语义。

除此之外，不要把普通工程问题抛给用户。

## 30. 最终交付内容

最终必须交付：

1. 完整项目代码。
2. 简洁配置文件。
3. README。
4. 详细设计文档。
5. Dataset 实现。
6. 多 train root 加载。
7. 单 val root 加载。
8. YOLO bbox parser。
9. `label_masks.json` 应用逻辑。
10. `annotations.jsonl` 元信息读取。
11. FAST-style 预处理算子。
12. target 生成。
13. SubFastNet 模型。
14. TextNet-style backbone。
15. FAST-like region/kernel heads。
16. loss。
17. metrics。
18. Burn TUI 训练。
19. validation。
20. inference。
21. benchmark。
22. checkpoint / resume。
23. tests。
24. 错误样本分析输出。
25. summary 和 metrics 输出。

最终项目必须能通过以下基本流程：

```bash
cargo run --release -- inspect-dataset --config configs/train.toml
cargo run --release -- train --config configs/train.toml
cargo run --release -- validate --config configs/train.toml --checkpoint outputs/subfastnet_tiny/best
cargo run --release -- infer --config configs/train.toml --checkpoint outputs/subfastnet_tiny/best --image sample.jpg
cargo run --release -- benchmark --config configs/train.toml --checkpoint outputs/subfastnet_tiny/best
```

## 31. 核心判断标准

项目是否合格，以以下标准判断：

1. 是否真正是字幕区域检测，而不是 OCR。
2. 是否正确支持多个训练 root。
3. 是否正确支持单验证 root。
4. 是否正确解析 YOLO bbox。
5. 是否强制应用 `label_masks.json`。
6. 是否读取并使用 `annotations.jsonl`。
7. 是否使用 SubFastNet 命名。
8. 是否以 TextNet / FAST-like 作为默认架构。
9. 是否使用 FAST-style 预处理。
10. 是否生成 `gt_instance`、`gt_text`、`gt_kernel`、`training_mask`。
11. 是否输出像素坐标 bbox。
12. 是否支持 CPU / CUDA / WGPU。
13. 是否接入 Burn TUI。
14. 是否有验证集指标。
15. 是否有 FPS / latency benchmark。
16. 是否有 checkpoint / resume。
17. 是否有完整文档。
18. 是否没有把需求错误扩展成 OCR 或通用文本检测。
