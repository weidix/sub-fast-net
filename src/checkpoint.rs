use std::{fs, path::Path};

use anyhow::{Context, Result};
use burn::{
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::Backend,
};
use serde::{Deserialize, Serialize};

use crate::{config::TrainConfig, model::SubFastNet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub epoch: usize,
    pub step: usize,
    pub best_f1: f32,
    #[serde(default)]
    pub learning_rate: f32,
    #[serde(default)]
    pub scheduler_epoch: usize,
    pub config: TrainConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerState {
    pub epoch: usize,
    pub step: usize,
    pub learning_rate: f32,
    pub gamma: f32,
}

pub fn save_checkpoint<B: Backend>(
    path: impl AsRef<Path>,
    model: SubFastNet<B>,
    meta: &CheckpointMeta,
) -> Result<()> {
    let path = path.as_ref();
    fs::create_dir_all(path)?;
    fs::write(
        path.join("checkpoint.json"),
        serde_json::to_string_pretty(meta)?,
    )?;
    model
        .save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::new(),
        )
        .map_err(|err| anyhow::anyhow!(err))
        .with_context(|| format!("failed to save model record {}", path.display()))?;
    save_scheduler_state(
        path,
        &SchedulerState {
            epoch: meta.scheduler_epoch,
            step: meta.step,
            learning_rate: meta.learning_rate,
            gamma: meta.config.scheduler_gamma,
        },
    )?;
    Ok(())
}

pub fn save_optimizer_record<O, B>(path: impl AsRef<Path>, optimizer: &O) -> Result<()>
where
    O: burn::optim::Optimizer<SubFastNet<B>, B>,
    B: burn::tensor::backend::AutodiffBackend,
{
    let path = path.as_ref();
    fs::create_dir_all(path)?;
    BinFileRecorder::<FullPrecisionSettings>::new()
        .record(optimizer.to_record(), path.join("optimizer"))
        .map_err(|err| anyhow::anyhow!(err))
        .with_context(|| format!("failed to save optimizer record {}", path.display()))?;
    Ok(())
}

pub fn load_optimizer_record<O, B>(path: impl AsRef<Path>, optimizer: O) -> Result<O>
where
    O: burn::optim::Optimizer<SubFastNet<B>, B>,
    B: burn::tensor::backend::AutodiffBackend,
{
    let path = path.as_ref();
    let optimizer_path = path.join("optimizer");
    if !optimizer_path.exists() && !optimizer_path.with_extension("bin").exists() {
        return Ok(optimizer);
    }
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load(optimizer_path, &B::Device::default())
        .map_err(|err| anyhow::anyhow!(err))
        .with_context(|| format!("failed to load optimizer record {}", path.display()))?;
    Ok(optimizer.load_record(record))
}

pub fn save_scheduler_state(path: impl AsRef<Path>, state: &SchedulerState) -> Result<()> {
    let path = path.as_ref();
    fs::write(
        path.join("scheduler.json"),
        serde_json::to_string_pretty(state)?,
    )?;
    Ok(())
}

pub fn load_scheduler_state(path: impl AsRef<Path>) -> Result<Option<SchedulerState>> {
    let path = path.as_ref().join("scheduler.json");
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read scheduler state {}", path.display()))?;
    Ok(Some(serde_json::from_str(&text)?))
}

pub fn load_checkpoint_meta(path: impl AsRef<Path>) -> Result<CheckpointMeta> {
    let path = path.as_ref();
    let text = fs::read_to_string(path.join("checkpoint.json"))
        .with_context(|| format!("failed to read checkpoint {}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

pub fn load_checkpoint_model<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(SubFastNet<B>, CheckpointMeta)> {
    let path = path.as_ref();
    let meta = load_checkpoint_meta(path)?;
    let model = SubFastNet::<B>::new(meta.config.model_variant, device)
        .load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .map_err(|err| anyhow::anyhow!(err))
        .with_context(|| format!("failed to load model record {}", path.display()))?;
    Ok((model, meta))
}

pub fn model_artifact_size_bytes(path: impl AsRef<Path>) -> Result<Option<u64>> {
    let path = path.as_ref();
    let model_bin = path.join("model.bin");
    let model_record = path.join("model");
    if model_bin.is_file() {
        return Ok(Some(fs::metadata(model_bin)?.len()));
    }
    if model_record.is_file() {
        return Ok(Some(fs::metadata(model_record)?.len()));
    }
    Ok(None)
}
