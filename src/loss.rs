use serde::Serialize;

use burn::tensor::{
    Bool, BoolStore, DType, FloatDType, Tensor, TensorData,
    activation::{log_sigmoid, sigmoid},
    backend::Backend,
};

use crate::model::{ModelOutput, sigmoid as sigmoid_scalar};

#[derive(Debug, Clone, Default, Serialize)]
pub struct LossBreakdown {
    pub total_loss: f32,
    pub region_bce_loss: f32,
    pub region_dice_loss: f32,
    pub kernel_bce_loss: f32,
    pub kernel_dice_loss: f32,
    pub bbox_loss: f32,
    pub ignored_area_ratio: f32,
    pub positive_region_ratio: f32,
    pub positive_kernel_ratio: f32,
}

pub struct TensorLossBreakdown<B: Backend> {
    pub total_loss: Tensor<B, 1>,
    pub region_bce_loss: Tensor<B, 1>,
    pub region_dice_loss: Tensor<B, 1>,
    pub kernel_bce_loss: Tensor<B, 1>,
    pub kernel_dice_loss: Tensor<B, 1>,
    pub ignored_area_ratio: Tensor<B, 1>,
    pub positive_region_ratio: Tensor<B, 1>,
    pub positive_kernel_ratio: Tensor<B, 1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossComponentSelection {
    pub region_bce: bool,
    pub kernel_bce: bool,
    pub region_dice: bool,
    pub kernel_dice: bool,
}

impl LossComponentSelection {
    pub const ALL: Self = Self {
        region_bce: true,
        kernel_bce: true,
        region_dice: true,
        kernel_dice: true,
    };
}

#[derive(Debug, Clone)]
pub struct LossTensorCache<B: Backend> {
    device: B::Device,
    dtype: DType,
    one: Tensor<B, 1>,
    two: Tensor<B, 1>,
    shaped: Option<LossShapedTensorCache<B>>,
}

#[derive(Debug, Clone)]
struct LossShapedTensorCache<B: Backend> {
    device: B::Device,
    dtype: DType,
    shape: [usize; 4],
    one: Tensor<B, 4>,
}

impl<B: Backend> LossTensorCache<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            device: device.clone(),
            dtype: DType::F32,
            one: cached_scalar(1.0, device),
            two: cached_scalar(2.0, device),
            shaped: None,
        }
    }

    fn ensure_device_and_dtype(&mut self, device: &B::Device, dtype: DType) {
        if self.device == *device && self.dtype == dtype {
            return;
        }
        self.device = device.clone();
        self.dtype = dtype;
        self.one = cached_scalar(1.0, device);
        self.two = cached_scalar(2.0, device);
        self.shaped = None;
    }

    fn ensure_shape(&mut self, shape: [usize; 4], device: &B::Device) {
        self.ensure_device_and_dtype(device, DType::F32);
        if self.shaped.as_ref().is_some_and(|cached| {
            cached.shape == shape && cached.device == *device && cached.dtype == DType::F32
        }) {
            return;
        }
        self.shaped = Some(LossShapedTensorCache {
            device: device.clone(),
            dtype: DType::F32,
            shape,
            one: cached_full(shape, 1.0, device),
        });
    }

    fn one(&self) -> Tensor<B, 1> {
        self.one.clone()
    }

    fn two(&self) -> Tensor<B, 1> {
        self.two.clone()
    }

    fn shaped_one(&self) -> Tensor<B, 4> {
        self.shaped
            .as_ref()
            .expect("loss tensor cache shape should be initialized")
            .one
            .clone()
    }
}

fn cached_scalar<B: Backend>(value: f32, device: &B::Device) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(vec![value], [1]), (device, DType::F32)).detach()
}

fn cached_full<B: Backend>(shape: [usize; 4], value: f32, device: &B::Device) -> Tensor<B, 4> {
    Tensor::full(shape, value, (device, DType::F32)).detach()
}

impl<B: Backend> TensorLossBreakdown<B> {
    pub fn total_loss_is_finite(&self) -> bool {
        bool_scalar_tensor_value(self.total_loss.clone().is_finite().all())
    }

    pub fn to_cpu_loss_breakdown(&self) -> LossBreakdown {
        LossBreakdown {
            total_loss: scalar_tensor_value(self.total_loss.clone()),
            region_bce_loss: scalar_tensor_value(self.region_bce_loss.clone()),
            region_dice_loss: scalar_tensor_value(self.region_dice_loss.clone()),
            kernel_bce_loss: scalar_tensor_value(self.kernel_bce_loss.clone()),
            kernel_dice_loss: scalar_tensor_value(self.kernel_dice_loss.clone()),
            bbox_loss: 0.0,
            ignored_area_ratio: scalar_tensor_value(self.ignored_area_ratio.clone()),
            positive_region_ratio: scalar_tensor_value(self.positive_region_ratio.clone()),
            positive_kernel_ratio: scalar_tensor_value(self.positive_kernel_ratio.clone()),
        }
    }
}

pub fn compute_loss(
    output: &crate::model::CpuModelOutput,
    gt_texts: &[Vec<f32>],
    gt_kernels: &[Vec<f32>],
    training_masks: &[Vec<f32>],
) -> LossBreakdown {
    let region_bce = masked_bce(&output.text_region_logits, gt_texts, training_masks);
    let kernel_bce = masked_bce(&output.kernel_logits, gt_kernels, training_masks);
    let region_dice = masked_dice(&output.text_region_logits, gt_texts, training_masks);
    let kernel_dice = masked_dice(&output.kernel_logits, gt_kernels, training_masks);
    let mut ignored: f32 = 0.0;
    let mut pixels: f32 = 0.0;
    let mut region_pos: f32 = 0.0;
    let mut kernel_pos: f32 = 0.0;
    for ((text, kernel), mask) in gt_texts.iter().zip(gt_kernels).zip(training_masks) {
        for ((text_value, kernel_value), mask_value) in text.iter().zip(kernel).zip(mask) {
            pixels += 1.0;
            if *mask_value <= 0.0 {
                ignored += 1.0;
            } else {
                region_pos += *text_value;
                kernel_pos += *kernel_value;
            }
        }
    }
    let valid = (pixels - ignored).max(1.0);
    LossBreakdown {
        total_loss: region_bce + region_dice + kernel_bce + kernel_dice,
        region_bce_loss: region_bce,
        region_dice_loss: region_dice,
        kernel_bce_loss: kernel_bce,
        kernel_dice_loss: kernel_dice,
        bbox_loss: 0.0,
        ignored_area_ratio: ignored / pixels.max(1.0),
        positive_region_ratio: region_pos / valid,
        positive_kernel_ratio: kernel_pos / valid,
    }
}

fn masked_bce(logits: &[Vec<f32>], targets: &[Vec<f32>], masks: &[Vec<f32>]) -> f32 {
    let mut loss = 0.0;
    let mut count: f32 = 0.0;
    for ((logit, target), mask) in logits.iter().zip(targets).zip(masks) {
        for ((logit_value, target_value), mask_value) in logit.iter().zip(target).zip(mask) {
            if *mask_value <= 0.0 {
                continue;
            }
            let p = sigmoid_scalar(*logit_value).clamp(1e-6, 1.0 - 1e-6);
            loss += -(*target_value * p.ln() + (1.0 - *target_value) * (1.0 - p).ln());
            count += 1.0;
        }
    }
    loss / count.max(1.0)
}

fn masked_dice(logits: &[Vec<f32>], targets: &[Vec<f32>], masks: &[Vec<f32>]) -> f32 {
    let mut intersection = 0.0;
    let mut pred_sum = 0.0;
    let mut target_sum = 0.0;
    for ((logit, target), mask) in logits.iter().zip(targets).zip(masks) {
        for ((logit_value, target_value), mask_value) in logit.iter().zip(target).zip(mask) {
            if *mask_value <= 0.0 {
                continue;
            }
            let pred = sigmoid_scalar(*logit_value);
            intersection += pred * *target_value;
            pred_sum += pred;
            target_sum += *target_value;
        }
    }
    1.0 - (2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0)
}

pub fn compute_tensor_loss<B: Backend>(
    output: &ModelOutput<B>,
    gt_text: Tensor<B, 4>,
    gt_kernel: Tensor<B, 4>,
    training_mask: Tensor<B, 4>,
) -> Tensor<B, 1> {
    compute_tensor_loss_breakdown(output, gt_text, gt_kernel, training_mask).total_loss
}

pub fn compute_tensor_loss_breakdown<B: Backend>(
    output: &ModelOutput<B>,
    gt_text: Tensor<B, 4>,
    gt_kernel: Tensor<B, 4>,
    training_mask: Tensor<B, 4>,
) -> TensorLossBreakdown<B> {
    let mut cache = LossTensorCache::new(&output.text_region_logits.device());
    compute_tensor_loss_breakdown_cached(output, gt_text, gt_kernel, training_mask, &mut cache)
}

pub fn compute_tensor_loss_breakdown_cached<B: Backend>(
    output: &ModelOutput<B>,
    gt_text: Tensor<B, 4>,
    gt_kernel: Tensor<B, 4>,
    training_mask: Tensor<B, 4>,
    cache: &mut LossTensorCache<B>,
) -> TensorLossBreakdown<B> {
    compute_tensor_loss_breakdown_cached_with_selection(
        output,
        gt_text,
        gt_kernel,
        training_mask,
        cache,
        LossComponentSelection::ALL,
    )
}

pub fn compute_tensor_loss_breakdown_cached_with_selection<B: Backend>(
    output: &ModelOutput<B>,
    gt_text: Tensor<B, 4>,
    gt_kernel: Tensor<B, 4>,
    training_mask: Tensor<B, 4>,
    cache: &mut LossTensorCache<B>,
    selection: LossComponentSelection,
) -> TensorLossBreakdown<B> {
    let output = ModelOutput {
        text_region_logits: output.text_region_logits.clone().cast(FloatDType::F32),
        kernel_logits: output.kernel_logits.clone().cast(FloatDType::F32),
        width: output.width,
        height: output.height,
    };
    let gt_text = gt_text.cast(FloatDType::F32);
    let gt_kernel = gt_kernel.cast(FloatDType::F32);
    let training_mask = training_mask.cast(FloatDType::F32);
    cache.ensure_shape(
        output.text_region_logits.dims(),
        &output.text_region_logits.device(),
    );
    let region_bce = masked_bce_tensor(
        output.text_region_logits.clone(),
        gt_text.clone(),
        training_mask.clone(),
        cache,
    );
    let kernel_bce = masked_bce_tensor(
        output.kernel_logits.clone(),
        gt_kernel.clone(),
        training_mask.clone(),
        cache,
    );
    let region_dice = masked_dice_tensor(
        output.text_region_logits.clone(),
        gt_text.clone(),
        training_mask.clone(),
        cache,
    );
    let kernel_dice = masked_dice_tensor(
        output.kernel_logits.clone(),
        gt_kernel.clone(),
        training_mask.clone(),
        cache,
    );
    let ratio_breakdown = ratio_breakdown_tensor(gt_text, gt_kernel, training_mask, cache);
    let region_bce_loss = region_bce.clone().detach();
    let region_dice_loss = region_dice.clone().detach();
    let kernel_bce_loss = kernel_bce.clone().detach();
    let kernel_dice_loss = kernel_dice.clone().detach();
    let mut total_loss = cache.one() * 0.0;
    if selection.region_bce {
        total_loss = total_loss + region_bce;
    }
    if selection.kernel_bce {
        total_loss = total_loss + kernel_bce;
    }
    if selection.region_dice {
        total_loss = total_loss + region_dice;
    }
    if selection.kernel_dice {
        total_loss = total_loss + kernel_dice;
    }
    TensorLossBreakdown {
        total_loss,
        region_bce_loss,
        region_dice_loss,
        kernel_bce_loss,
        kernel_dice_loss,
        ignored_area_ratio: ratio_breakdown.ignored_area_ratio.detach(),
        positive_region_ratio: ratio_breakdown.positive_region_ratio.detach(),
        positive_kernel_ratio: ratio_breakdown.positive_kernel_ratio.detach(),
    }
}

struct TensorRatioBreakdown<B: Backend> {
    ignored_area_ratio: Tensor<B, 1>,
    positive_region_ratio: Tensor<B, 1>,
    positive_kernel_ratio: Tensor<B, 1>,
}

fn masked_bce_tensor<B: Backend>(
    logits: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    mask: Tensor<B, 4>,
    cache: &LossTensorCache<B>,
) -> Tensor<B, 1> {
    let inverse_targets = cache.shaped_one() - targets;
    let bce = logits.clone() * inverse_targets - log_sigmoid(logits);
    let masked = bce * mask.clone();
    masked.sum() / mask.sum().max_pair(cache.one())
}

fn masked_dice_tensor<B: Backend>(
    logits: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    mask: Tensor<B, 4>,
    cache: &LossTensorCache<B>,
) -> Tensor<B, 1> {
    let probs = sigmoid(logits) * mask.clone();
    let targets = targets * mask;
    let intersection = (probs.clone() * targets.clone()).sum();
    let denom = probs.sum() + targets.sum();
    cache.one() - ((intersection * cache.two() + cache.one()) / (denom + cache.one()))
}

fn ratio_breakdown_tensor<B: Backend>(
    gt_text: Tensor<B, 4>,
    gt_kernel: Tensor<B, 4>,
    training_mask: Tensor<B, 4>,
    cache: &LossTensorCache<B>,
) -> TensorRatioBreakdown<B> {
    let pixel_count = training_mask.dims().iter().product::<usize>() as f32;
    let valid_count_raw = training_mask.clone().sum();
    let valid_count = valid_count_raw.clone().max_pair(cache.one());
    let ignored_count = pixel_count - valid_count_raw;
    TensorRatioBreakdown {
        ignored_area_ratio: ignored_count / pixel_count.max(1.0),
        positive_region_ratio: (gt_text * training_mask.clone()).sum() / valid_count.clone(),
        positive_kernel_ratio: (gt_kernel * training_mask).sum() / valid_count,
    }
}

fn scalar_tensor_value<B: Backend>(tensor: Tensor<B, 1>) -> f32 {
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .expect("scalar tensor should be f32");
    values
        .first()
        .copied()
        .expect("scalar tensor should contain one value")
}

fn bool_scalar_tensor_value<B: Backend>(tensor: Tensor<B, 1, Bool>) -> bool {
    let data = tensor.into_data();
    match data.dtype {
        DType::Bool(BoolStore::Native) => data
            .to_vec::<bool>()
            .expect("scalar tensor should be native bool")
            .first()
            .copied()
            .expect("scalar tensor should contain one value"),
        DType::Bool(BoolStore::U8) => {
            data.to_vec::<u8>()
                .expect("scalar tensor should be u8 bool")
                .first()
                .copied()
                .expect("scalar tensor should contain one value")
                != 0
        }
        DType::Bool(BoolStore::U32) => {
            data.to_vec::<u32>()
                .expect("scalar tensor should be u32 bool")
                .first()
                .copied()
                .expect("scalar tensor should contain one value")
                != 0
        }
        dtype => panic!("scalar tensor should be bool, got {dtype:?}"),
    }
}
