use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::{
    backend::{infer_with_configured_backend_and_roi, validate_with_configured_backend},
    benchmark::benchmark,
    config::TrainConfig,
    dataset::SubtitleDataset,
    infer::{InferRoi, write_inference_json},
    train::train,
    validate::write_validation_outputs,
};

#[derive(Debug, Parser)]
#[command(name = "sub-fast-net")]
#[command(about = "SubFastNet subtitle region detector")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Train {
        #[arg(long)]
        config: String,
    },
    Validate {
        #[arg(long)]
        config: String,
        #[arg(long)]
        checkpoint: String,
    },
    Infer {
        #[arg(long)]
        config: String,
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        image: String,
        #[arg(long, num_args = 2)]
        roi_offset: Option<Vec<i32>>,
        #[arg(long, num_args = 2)]
        frame_size: Option<Vec<u32>>,
    },
    InspectDataset {
        #[arg(long)]
        config: String,
    },
    Benchmark {
        #[arg(long)]
        config: String,
        #[arg(long)]
        checkpoint: String,
    },
}

pub fn run() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Train { config } => {
            let config = TrainConfig::from_path(config)?;
            let summary = train(&config)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Validate { config, checkpoint } => {
            let config = TrainConfig::from_path(config)?;
            let summary = validate_with_configured_backend(&config, &checkpoint)?;
            write_validation_outputs(&config, &summary)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Infer {
            config,
            checkpoint,
            image,
            roi_offset,
            frame_size,
        } => {
            let config = TrainConfig::from_path(config)?;
            let roi = roi_offset.map(|offset| InferRoi {
                offset: [offset[0], offset[1]],
                frame_size: frame_size.map(|size| [size[0], size[1]]),
            });
            let output = infer_with_configured_backend_and_roi(&config, &checkpoint, &image, roi)?;
            write_inference_json(&output)?;
        }
        Command::InspectDataset { config } => {
            let config = TrainConfig::from_path(config)?;
            let train = SubtitleDataset::from_train_config(&config)?;
            let val = SubtitleDataset::from_val_config(&config)?;
            let report = serde_json::json!({
                "train": train.inspect(),
                "val": val.inspect(),
            });
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Benchmark { config, checkpoint } => {
            let config = TrainConfig::from_path(config)?;
            let summary = benchmark(&config, &checkpoint)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
    }
    Ok(())
}
