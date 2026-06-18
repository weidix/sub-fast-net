use burn::{
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{
        FloatDType, Tensor,
        activation::relu,
        backend::Backend,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};
use serde::{Deserialize, Serialize};

use crate::{config::ModelVariant, preprocess::PixelBox};

#[derive(Module, Debug)]
pub struct SubFastNet<B: Backend> {
    stem: TextNetStem<B>,
    stride4: TextNetStage<B>,
    stride8: TextNetStage<B>,
    stride16: TextNetStage<B>,
    align4: Conv2d<B>,
    align8: Conv2d<B>,
    align16: Conv2d<B>,
    fusion_refine: TextNetBlock<B>,
    region_head: DetectionHead<B>,
    kernel_head: DetectionHead<B>,
    #[module(skip)]
    variant: ModelVariant,
}

#[derive(Module, Debug)]
struct TextNetStem<B: Backend> {
    reduce: Conv2d<B>,
    refine: TextNetBlock<B>,
}

#[derive(Module, Debug)]
struct TextNetStage<B: Backend> {
    downsample: Conv2d<B>,
    block_a: TextNetBlock<B>,
    block_b: TextNetBlock<B>,
}

#[derive(Module, Debug)]
struct TextNetBlock<B: Backend> {
    vertical: Conv2d<B>,
    horizontal: Conv2d<B>,
    pointwise: Conv2d<B>,
}

#[derive(Module, Debug)]
struct DetectionHead<B: Backend> {
    refine: Conv2d<B>,
    logits: Conv2d<B>,
}

#[derive(Debug, Clone)]
pub struct ModelOutput<B: Backend> {
    pub text_region_logits: Tensor<B, 4>,
    pub kernel_logits: Tensor<B, 4>,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone)]
pub struct CpuModelOutput {
    pub text_region_logits: Vec<Vec<f32>>,
    pub kernel_logits: Vec<Vec<f32>>,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Copy)]
struct VariantChannels {
    stem: usize,
    stride4: usize,
    stride8: usize,
    stride16: usize,
    fusion: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubFastNetArchitecture {
    pub feature_strides: [usize; 3],
    pub output_stride: usize,
    pub detection_head_count: usize,
    pub region_head_enabled: bool,
    pub kernel_head_enabled: bool,
}

impl<B: Backend> SubFastNet<B> {
    pub fn new(variant: ModelVariant, device: &B::Device) -> Self {
        let channels = variant_channels(variant);
        Self {
            stem: TextNetStem::new(channels.stem, device),
            stride4: TextNetStage::new(channels.stem, channels.stride4, device),
            stride8: TextNetStage::new(channels.stride4, channels.stride8, device),
            stride16: TextNetStage::new(channels.stride8, channels.stride16, device),
            align4: conv_same([channels.stride4, channels.fusion], [1, 1], [1, 1], device),
            align8: conv_same([channels.stride8, channels.fusion], [1, 1], [1, 1], device),
            align16: conv_same([channels.stride16, channels.fusion], [1, 1], [1, 1], device),
            fusion_refine: TextNetBlock::new(channels.fusion, channels.fusion, device),
            region_head: DetectionHead::new(channels.fusion, device),
            kernel_head: DetectionHead::new(channels.fusion, device),
            variant,
        }
    }

    pub fn forward(&self, images: Tensor<B, 4>) -> ModelOutput<B> {
        let x = self.stem.forward(images);
        let p2 = self.stride4.forward(x);
        let p3 = self.stride8.forward(p2.clone());
        let p4 = self.stride16.forward(p3.clone());
        let shape = p2.shape();
        let target_h = shape.dims::<4>()[2];
        let target_w = shape.dims::<4>()[3];
        let f2 = self.align4.forward(p2);
        let f3 = nearest_upsample_to(self.align8.forward(p3), target_h, target_w);
        let f4 = nearest_upsample_to(self.align16.forward(p4), target_h, target_w);
        let fused = self.fusion_refine.forward((f2 + f3 + f4) / 3.0);
        let text_region_logits = self.region_head.forward(fused.clone());
        let kernel_logits = self.kernel_head.forward(fused);
        ModelOutput {
            text_region_logits,
            kernel_logits,
            width: target_w,
            height: target_h,
        }
    }

    pub fn forward_head_only_backward(&self, images: Tensor<B, 4>) -> ModelOutput<B> {
        let x = self.stem.forward(images);
        let p2 = self.stride4.forward(x);
        let p3 = self.stride8.forward(p2.clone());
        let p4 = self.stride16.forward(p3.clone());
        let shape = p2.shape();
        let target_h = shape.dims::<4>()[2];
        let target_w = shape.dims::<4>()[3];
        let f2 = self.align4.forward(p2);
        let f3 = nearest_upsample_to(self.align8.forward(p3), target_h, target_w);
        let f4 = nearest_upsample_to(self.align16.forward(p4), target_h, target_w);
        let fused = self.fusion_refine.forward((f2 + f3 + f4) / 3.0).detach();
        let text_region_logits = self.region_head.forward(fused.clone());
        let kernel_logits = self.kernel_head.forward(fused);
        ModelOutput {
            text_region_logits,
            kernel_logits,
            width: target_w,
            height: target_h,
        }
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn architecture(&self) -> SubFastNetArchitecture {
        architecture_spec()
    }
}

impl<B: Backend> TextNetStem<B> {
    fn new(out_channels: usize, device: &B::Device) -> Self {
        Self {
            reduce: conv_same([3, out_channels], [3, 3], [2, 2], device),
            refine: TextNetBlock::new(out_channels, out_channels, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.refine.forward(relu(self.reduce.forward(x)))
    }
}

impl<B: Backend> TextNetStage<B> {
    fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        Self {
            downsample: conv_same([in_channels, out_channels], [3, 3], [2, 2], device),
            block_a: TextNetBlock::new(out_channels, out_channels, device),
            block_b: TextNetBlock::new(out_channels, out_channels, device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = relu(self.downsample.forward(x));
        let x = self.block_a.forward(x);
        self.block_b.forward(x)
    }
}

impl<B: Backend> TextNetBlock<B> {
    fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        Self {
            vertical: conv_same([in_channels, out_channels], [3, 1], [1, 1], device),
            horizontal: conv_same([out_channels, out_channels], [1, 3], [1, 1], device),
            pointwise: conv_same([out_channels, out_channels], [1, 1], [1, 1], device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = relu(self.vertical.forward(x));
        let x = relu(self.horizontal.forward(x));
        relu(self.pointwise.forward(x))
    }
}

impl<B: Backend> DetectionHead<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            refine: conv_same([channels, channels], [3, 3], [1, 1], device),
            logits: conv_same([channels, 1], [1, 1], [1, 1], device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.logits.forward(relu(self.refine.forward(x)))
    }
}

fn variant_channels(variant: ModelVariant) -> VariantChannels {
    match variant {
        ModelVariant::Tiny => VariantChannels {
            stem: 24,
            stride4: 36,
            stride8: 56,
            stride16: 80,
            fusion: 56,
        },
        ModelVariant::Small => VariantChannels {
            stem: 32,
            stride4: 48,
            stride8: 72,
            stride16: 96,
            fusion: 64,
        },
        ModelVariant::Base => VariantChannels {
            stem: 40,
            stride4: 64,
            stride8: 96,
            stride16: 128,
            fusion: 80,
        },
    }
}

pub fn architecture_spec() -> SubFastNetArchitecture {
    SubFastNetArchitecture {
        feature_strides: [4, 8, 16],
        output_stride: 4,
        detection_head_count: 2,
        region_head_enabled: true,
        kernel_head_enabled: true,
    }
}

fn conv_same<B: Backend>(
    channels: [usize; 2],
    kernel: [usize; 2],
    stride: [usize; 2],
    device: &B::Device,
) -> Conv2d<B> {
    Conv2dConfig::new(channels, kernel)
        .with_stride(stride)
        .with_padding(PaddingConfig2d::Same)
        .init(device)
}

fn nearest_upsample_to<B: Backend>(
    tensor: Tensor<B, 4>,
    target_h: usize,
    target_w: usize,
) -> Tensor<B, 4> {
    interpolate(
        tensor,
        [target_h, target_w],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

pub fn parameter_count_estimate(variant: ModelVariant) -> usize {
    let c = variant_channels(variant);
    let stem = conv_params(3, c.stem, 3, 3) + textnet_block_params(c.stem, c.stem);
    let stride4 = stage_params(c.stem, c.stride4);
    let stride8 = stage_params(c.stride4, c.stride8);
    let stride16 = stage_params(c.stride8, c.stride16);
    let align = conv_params(c.stride4, c.fusion, 1, 1)
        + conv_params(c.stride8, c.fusion, 1, 1)
        + conv_params(c.stride16, c.fusion, 1, 1);
    let fusion = textnet_block_params(c.fusion, c.fusion);
    let heads = 2 * (conv_params(c.fusion, c.fusion, 3, 3) + conv_params(c.fusion, 1, 1, 1));
    stem + stride4 + stride8 + stride16 + align + fusion + heads
}

pub fn serialized_size_bytes_estimate(variant: ModelVariant) -> usize {
    parameter_count_estimate(variant) * 4
}

fn stage_params(in_channels: usize, out_channels: usize) -> usize {
    conv_params(in_channels, out_channels, 3, 3)
        + textnet_block_params(out_channels, out_channels)
        + textnet_block_params(out_channels, out_channels)
}

fn textnet_block_params(in_channels: usize, out_channels: usize) -> usize {
    conv_params(in_channels, out_channels, 3, 1)
        + conv_params(out_channels, out_channels, 1, 3)
        + conv_params(out_channels, out_channels, 1, 1)
}

fn conv_params(in_channels: usize, out_channels: usize, kernel_h: usize, kernel_w: usize) -> usize {
    in_channels * out_channels * kernel_h * kernel_w + out_channels
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

impl From<(PixelBox, f32)> for DetectionBox {
    fn from((bbox, confidence): (PixelBox, f32)) -> Self {
        Self {
            x1: bbox.x1,
            y1: bbox.y1,
            x2: bbox.x2,
            y2: bbox.y2,
            confidence,
        }
    }
}

pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub fn output_to_cpu<B: Backend>(output: ModelOutput<B>) -> CpuModelOutput {
    let region_shape = output.text_region_logits.shape();
    let kernel_shape = output.kernel_logits.shape();
    let batch = region_shape.dims::<4>()[0];
    let plane = output.width * output.height;
    let region_values = output
        .text_region_logits
        .cast(FloatDType::F32)
        .into_data()
        .to_vec::<f32>()
        .expect("region logits should be f32");
    let kernel_values = output
        .kernel_logits
        .cast(FloatDType::F32)
        .into_data()
        .to_vec::<f32>()
        .expect("kernel logits should be f32");
    debug_assert_eq!(region_shape.dims::<4>(), kernel_shape.dims::<4>());
    CpuModelOutput {
        text_region_logits: region_values
            .chunks(plane)
            .take(batch)
            .map(|chunk| chunk.to_vec())
            .collect(),
        kernel_logits: kernel_values
            .chunks(plane)
            .take(batch)
            .map(|chunk| chunk.to_vec())
            .collect(),
        width: output.width,
        height: output.height,
    }
}
