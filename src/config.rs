use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainConfig {
    pub experiment_name: String,
    pub output_dir: String,
    pub seed: u64,
    pub backend: BackendKind,
    pub train_roots: Vec<String>,
    pub val_root: String,
    pub model_variant: ModelVariant,
    pub input_size: u32,
    pub input_width: Option<u32>,
    pub input_height: Option<u32>,
    pub short_size: u32,
    pub alignment: u32,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub optimizer: OptimizerKind,
    pub gradient_accumulation_steps: usize,
    pub validation_interval: usize,
    pub checkpoint_interval: usize,
    pub log_interval: usize,
    pub prefetch_batches: usize,
    pub threshold_region: f32,
    pub threshold_kernel: f32,
    pub max_detection_width_ratio: f32,
    pub iou_threshold: f32,
    pub pooling_size: usize,
    pub shrink_kernel_scale: f32,
    pub min_kernel_width: u32,
    pub min_kernel_height: u32,
    pub scale_min: f32,
    pub scale_max: f32,
    pub aspect_min: f32,
    pub aspect_max: f32,
    pub random_horizontal_flip: bool,
    pub flip_prob: f32,
    pub random_rotate: bool,
    pub rotate_angle: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
    pub gaussian_blur: bool,
    pub gaussian_blur_prob: f32,
    pub scheduler_gamma: f32,
    pub augment_enabled: bool,
    pub strict_dataset: bool,
    pub tui_enabled: bool,
    pub profiling_enabled: bool,
    pub profiling_ablation: ProfilingAblation,
    pub resume: String,
    pub max_train_samples: Option<usize>,
    pub max_val_samples: Option<usize>,
    pub train_empty_sample_ratio: Option<f32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Cpu,
    Cuda,
    Wgpu,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelVariant {
    Tiny,
    Small,
    Base,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerKind {
    Adam,
    Sgd,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProfilingAblation {
    #[default]
    Normal,
    ForwardOnly,
    ForwardLossOnly,
    BackwardOnly,
    RegionBceOnly,
    KernelBceOnly,
    RegionDiceOnly,
    KernelDiceOnly,
    BceOnly,
    DiceOnly,
    RegionOnly,
    KernelOnly,
    DummyScalarBackward,
    HeadOnlyBackward,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            experiment_name: "subfastnet_tiny".to_string(),
            output_dir: "outputs/subfastnet_tiny".to_string(),
            seed: 42,
            backend: BackendKind::Wgpu,
            train_roots: vec![
                "data/generated_samples1".to_string(),
                "data/generated_samples2".to_string(),
                "data/generated_samples3".to_string(),
                "data/generated_samples4".to_string(),
                "data/mixed_subtitle_samples".to_string(),
            ],
            val_root: "data/validation_samples".to_string(),
            model_variant: ModelVariant::Tiny,
            input_size: 640,
            input_width: None,
            input_height: None,
            short_size: 640,
            alignment: 32,
            batch_size: 16,
            epochs: 100,
            learning_rate: 0.001,
            optimizer: OptimizerKind::Adam,
            gradient_accumulation_steps: 1,
            validation_interval: 1,
            checkpoint_interval: 1,
            log_interval: 50,
            prefetch_batches: 2,
            threshold_region: 0.5,
            threshold_kernel: 0.5,
            max_detection_width_ratio: 1.0,
            iou_threshold: 0.5,
            pooling_size: 9,
            shrink_kernel_scale: 0.1,
            min_kernel_width: 3,
            min_kernel_height: 3,
            scale_min: 0.7,
            scale_max: 1.3,
            aspect_min: 0.9,
            aspect_max: 1.1,
            random_horizontal_flip: true,
            flip_prob: 0.5,
            random_rotate: true,
            rotate_angle: 5.0,
            brightness: 0.125,
            contrast: 0.4,
            saturation: 0.4,
            hue: 0.1,
            gaussian_blur: true,
            gaussian_blur_prob: 0.5,
            scheduler_gamma: 1.0,
            augment_enabled: true,
            strict_dataset: false,
            tui_enabled: true,
            profiling_enabled: false,
            profiling_ablation: ProfilingAblation::Normal,
            resume: String::new(),
            max_train_samples: None,
            max_val_samples: None,
            train_empty_sample_ratio: None,
        }
    }
}

impl TrainConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let raw: toml::Value = toml::from_str(&text)
            .with_context(|| format!("failed to parse config {}", path.display()))?;
        if raw.get("mixed_precision").is_some() {
            bail!("mixed_precision has been removed; all training uses FP32");
        }
        let config: Self = toml::from_str(&text)
            .with_context(|| format!("failed to parse config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.train_roots.is_empty() {
            bail!("train_roots must contain at least one dataset root");
        }
        if self.val_root.trim().is_empty() {
            bail!("val_root must be configured");
        }
        if self.batch_size == 0 {
            bail!("batch_size must be greater than zero");
        }
        if self.prefetch_batches == 0 {
            bail!("prefetch_batches must be greater than zero");
        }
        if self.gradient_accumulation_steps == 0 {
            bail!("gradient_accumulation_steps must be greater than zero");
        }
        if self.input_width().min(self.input_height()) == 0 {
            bail!("input size must be greater than zero");
        }
        if self.alignment == 0 {
            bail!("alignment must be greater than zero");
        }
        if self.pooling_size == 0 || self.pooling_size.is_multiple_of(2) {
            bail!("pooling_size must be a positive odd number");
        }
        if self.scale_min <= 0.0 || self.scale_max < self.scale_min {
            bail!("scale_min/scale_max must define a positive range");
        }
        if !(0.0..=1.0).contains(&self.flip_prob) || !(0.0..=1.0).contains(&self.gaussian_blur_prob)
        {
            bail!("augmentation probabilities must be in [0, 1]");
        }
        if self.max_detection_width_ratio <= 0.0 || self.max_detection_width_ratio > 1.0 {
            bail!("max_detection_width_ratio must be in (0, 1]");
        }
        if let Some(ratio) = self.train_empty_sample_ratio
            && !(0.0..=1.0).contains(&ratio)
        {
            bail!("train_empty_sample_ratio must be in [0, 1]");
        }
        Ok(())
    }

    pub fn input_width(&self) -> u32 {
        self.input_width.unwrap_or(self.input_size)
    }

    pub fn input_height(&self) -> u32 {
        self.input_height.unwrap_or(self.input_size)
    }

    pub fn save_snapshot(&self) -> Result<()> {
        fs::create_dir_all(&self.output_dir)
            .with_context(|| format!("failed to create {}", self.output_dir))?;
        let text = toml::to_string_pretty(self)?;
        fs::write(
            Path::new(&self.output_dir).join("config.snapshot.toml"),
            text,
        )?;
        Ok(())
    }
}
