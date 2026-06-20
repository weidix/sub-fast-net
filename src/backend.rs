use anyhow::Result;
use burn::backend::{Autodiff, NdArray};

use crate::{
    benchmark::BenchmarkSummary,
    checkpoint::load_checkpoint_model,
    config::{BackendKind, TrainConfig},
    dataset::SubtitleDataset,
    infer::{InferRoi, InferenceOutput},
    train::TrainingSummary,
    validate::ValidationSummary,
};

pub type CpuBackend = NdArray<f32, i64>;
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;

#[cfg(feature = "backend-cuda")]
pub type CudaBackend = burn::backend::Cuda<f32, i32>;
#[cfg(feature = "backend-cuda")]
pub type CudaAutodiffBackend = Autodiff<CudaBackend>;

#[cfg(feature = "backend-wgpu")]
pub type WgpuBackend = burn::backend::Wgpu<f32, i32>;
#[cfg(feature = "backend-wgpu")]
pub type WgpuAutodiffBackend = Autodiff<WgpuBackend>;

pub fn train_with_configured_backend(config: &TrainConfig) -> Result<TrainingSummary> {
    match config.backend {
        BackendKind::Cpu => crate::train::train_backend::<CpuAutodiffBackend>(config),
        BackendKind::Cuda => train_cuda(config),
        BackendKind::Wgpu => train_wgpu(config),
    }
}

pub fn validate_with_configured_backend(
    config: &TrainConfig,
    checkpoint: &str,
) -> Result<ValidationSummary> {
    match config.backend {
        BackendKind::Cpu => validate_backend::<CpuBackend>(config, checkpoint),
        BackendKind::Cuda => validate_cuda(config, checkpoint),
        BackendKind::Wgpu => validate_wgpu(config, checkpoint),
    }
}

pub fn infer_with_configured_backend(
    config: &TrainConfig,
    checkpoint: &str,
    image_path: &str,
) -> Result<InferenceOutput> {
    infer_with_configured_backend_and_roi(config, checkpoint, image_path, None)
}

pub fn infer_with_configured_backend_and_roi(
    config: &TrainConfig,
    checkpoint: &str,
    image_path: &str,
    roi: Option<InferRoi>,
) -> Result<InferenceOutput> {
    match config.backend {
        BackendKind::Cpu => crate::infer::infer_image_backend_with_roi::<CpuBackend>(
            config, checkpoint, image_path, roi,
        ),
        BackendKind::Cuda => infer_cuda(config, checkpoint, image_path, roi),
        BackendKind::Wgpu => infer_wgpu(config, checkpoint, image_path, roi),
    }
}

pub fn benchmark_with_configured_backend(
    config: &TrainConfig,
    checkpoint: &str,
) -> Result<BenchmarkSummary> {
    match config.backend {
        BackendKind::Cpu => crate::benchmark::benchmark_backend::<CpuBackend>(config, checkpoint),
        BackendKind::Cuda => benchmark_cuda(config, checkpoint),
        BackendKind::Wgpu => benchmark_wgpu(config, checkpoint),
    }
}

fn validate_backend<B: burn::tensor::backend::Backend>(
    config: &TrainConfig,
    checkpoint: &str,
) -> Result<ValidationSummary> {
    let device = B::Device::default();
    let (model, _meta) = load_checkpoint_model::<B>(checkpoint, &device)?;
    let dataset = SubtitleDataset::from_val_config(config)?;
    crate::validate::validate_model(config, &dataset, &model)
}

#[cfg(feature = "backend-cuda")]
fn train_cuda(config: &TrainConfig) -> Result<TrainingSummary> {
    crate::train::train_backend::<CudaAutodiffBackend>(config)
}

#[cfg(not(feature = "backend-cuda"))]
fn train_cuda(_config: &TrainConfig) -> Result<TrainingSummary> {
    anyhow::bail!("backend=cuda requires building with feature backend-cuda")
}

#[cfg(feature = "backend-wgpu")]
fn train_wgpu(config: &TrainConfig) -> Result<TrainingSummary> {
    crate::train::train_backend::<WgpuAutodiffBackend>(config)
}

#[cfg(not(feature = "backend-wgpu"))]
fn train_wgpu(_config: &TrainConfig) -> Result<TrainingSummary> {
    anyhow::bail!("backend=wgpu requires building with feature backend-wgpu")
}

#[cfg(feature = "backend-cuda")]
fn validate_cuda(config: &TrainConfig, checkpoint: &str) -> Result<ValidationSummary> {
    validate_backend::<CudaBackend>(config, checkpoint)
}

#[cfg(not(feature = "backend-cuda"))]
fn validate_cuda(_config: &TrainConfig, _checkpoint: &str) -> Result<ValidationSummary> {
    anyhow::bail!("backend=cuda requires building with feature backend-cuda")
}

#[cfg(feature = "backend-wgpu")]
fn validate_wgpu(config: &TrainConfig, checkpoint: &str) -> Result<ValidationSummary> {
    validate_backend::<WgpuBackend>(config, checkpoint)
}

#[cfg(not(feature = "backend-wgpu"))]
fn validate_wgpu(_config: &TrainConfig, _checkpoint: &str) -> Result<ValidationSummary> {
    anyhow::bail!("backend=wgpu requires building with feature backend-wgpu")
}

#[cfg(feature = "backend-cuda")]
fn infer_cuda(
    config: &TrainConfig,
    checkpoint: &str,
    image_path: &str,
    roi: Option<InferRoi>,
) -> Result<InferenceOutput> {
    crate::infer::infer_image_backend_with_roi::<CudaBackend>(config, checkpoint, image_path, roi)
}

#[cfg(not(feature = "backend-cuda"))]
fn infer_cuda(
    _config: &TrainConfig,
    _checkpoint: &str,
    _image_path: &str,
    _roi: Option<InferRoi>,
) -> Result<InferenceOutput> {
    anyhow::bail!("backend=cuda requires building with feature backend-cuda")
}

#[cfg(feature = "backend-wgpu")]
fn infer_wgpu(
    config: &TrainConfig,
    checkpoint: &str,
    image_path: &str,
    roi: Option<InferRoi>,
) -> Result<InferenceOutput> {
    crate::infer::infer_image_backend_with_roi::<WgpuBackend>(config, checkpoint, image_path, roi)
}

#[cfg(not(feature = "backend-wgpu"))]
fn infer_wgpu(
    _config: &TrainConfig,
    _checkpoint: &str,
    _image_path: &str,
    _roi: Option<InferRoi>,
) -> Result<InferenceOutput> {
    anyhow::bail!("backend=wgpu requires building with feature backend-wgpu")
}

#[cfg(feature = "backend-cuda")]
fn benchmark_cuda(config: &TrainConfig, checkpoint: &str) -> Result<BenchmarkSummary> {
    crate::benchmark::benchmark_backend::<CudaBackend>(config, checkpoint)
}

#[cfg(not(feature = "backend-cuda"))]
fn benchmark_cuda(_config: &TrainConfig, _checkpoint: &str) -> Result<BenchmarkSummary> {
    anyhow::bail!("backend=cuda requires building with feature backend-cuda")
}

#[cfg(feature = "backend-wgpu")]
fn benchmark_wgpu(config: &TrainConfig, checkpoint: &str) -> Result<BenchmarkSummary> {
    crate::benchmark::benchmark_backend::<WgpuBackend>(config, checkpoint)
}

#[cfg(not(feature = "backend-wgpu"))]
fn benchmark_wgpu(_config: &TrainConfig, _checkpoint: &str) -> Result<BenchmarkSummary> {
    anyhow::bail!("backend=wgpu requires building with feature backend-wgpu")
}
