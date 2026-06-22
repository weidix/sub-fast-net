use std::{collections::HashMap, io::IsTerminal, sync::Arc};

use burn::data::dataloader::Progress;
use burn::train::{
    Interrupter,
    metric::{
        MetricAttributes, MetricEntry, MetricId, NumericAttributes, NumericEntry, SerializedEntry,
    },
    renderer::{
        MetricState, MetricsRenderer, ProgressType, TrainingProgress,
        tui::TuiMetricsRendererWrapper,
    },
};

use crate::{config::TrainConfig, train::TrainStepMetrics, validate::ValidationSummary};

pub struct BurnTui {
    renderer: Option<Box<dyn MetricsRenderer>>,
    metric_ids: HashMap<&'static str, MetricId>,
    epochs: usize,
}

impl BurnTui {
    pub fn new(config: &TrainConfig) -> Self {
        if !config.tui_enabled || !std::io::stdout().is_terminal() {
            return Self {
                renderer: None,
                metric_ids: HashMap::new(),
                epochs: config.epochs,
            };
        }

        let mut renderer: Box<dyn MetricsRenderer> =
            Box::new(TuiMetricsRendererWrapper::new(Interrupter::new(), None));
        let mut metric_ids = HashMap::new();
        for metric in TRAIN_METRICS {
            register_metric(&mut *renderer, &mut metric_ids, metric, None, false);
        }
        for metric in VALID_METRICS {
            register_metric(&mut *renderer, &mut metric_ids, metric, None, true);
        }
        Self {
            renderer: Some(renderer),
            metric_ids,
            epochs: config.epochs,
        }
    }

    pub fn is_active(&self) -> bool {
        self.renderer.is_some()
    }

    pub fn update_train(&mut self, metrics: &TrainStepMetrics) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        update_train_metric(
            &mut **renderer,
            &self.metric_ids,
            "loss",
            metrics.total_loss,
        );
        update_train_metric(
            &mut **renderer,
            &self.metric_ids,
            "samples/s",
            metrics.samples_per_second,
        );
        let progress = train_progress(metrics, self.epochs);
        renderer.render_train(progress, train_progress_indicators(metrics, self.epochs));
    }

    pub fn update_train_progress(
        &mut self,
        epoch: usize,
        step: usize,
        epoch_batch: usize,
        epoch_batches: usize,
        epoch_samples_processed: usize,
        epoch_samples_total: usize,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        renderer.render_train(
            train_progress_from_values(
                epoch,
                step,
                epoch_samples_processed,
                epoch_samples_total,
                self.epochs,
            ),
            train_progress_indicators_from_values(
                epoch,
                step,
                epoch_batch,
                epoch_batches,
                epoch_samples_processed,
                epoch_samples_total,
                self.epochs,
            ),
        );
    }

    pub fn update_valid(&mut self, epoch: usize, step: usize, validation: &ValidationSummary) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        update_valid_metric(
            &mut **renderer,
            &self.metric_ids,
            "loss",
            validation.val_loss,
        );
        update_valid_metric(&mut **renderer, &self.metric_ids, "f1", validation.f1);
        update_valid_metric(
            &mut **renderer,
            &self.metric_ids,
            "precision",
            validation.precision,
        );
        update_valid_metric(
            &mut **renderer,
            &self.metric_ids,
            "recall",
            validation.recall,
        );
        update_valid_metric(
            &mut **renderer,
            &self.metric_ids,
            "mean_iou",
            validation.mean_iou,
        );
        renderer.render_valid(
            valid_progress(epoch, step, self.epochs),
            vec![
                ProgressType::Detailed {
                    tag: "Epoch".to_string(),
                    progress: Progress {
                        items_processed: epoch,
                        items_total: self.epochs,
                    },
                },
                ProgressType::Value {
                    tag: "Iteration".to_string(),
                    value: step,
                },
            ],
        );
    }

    pub fn finish(&mut self) {
        if let Some(renderer) = self.renderer.as_mut() {
            let _ = renderer.on_train_end(None);
        }
    }
}

const TRAIN_METRICS: &[&str] = &["loss", "samples/s"];

const VALID_METRICS: &[&str] = &["loss", "f1", "precision", "recall", "mean_iou"];

fn register_metric(
    renderer: &mut dyn MetricsRenderer,
    metric_ids: &mut HashMap<&'static str, MetricId>,
    name: &'static str,
    unit: Option<&str>,
    higher_is_better: bool,
) {
    let id = MetricId::new(Arc::new(name.to_string()));
    renderer.register_metric(burn::train::metric::MetricDefinition {
        metric_id: id.clone(),
        name: name.to_string(),
        description: None,
        attributes: MetricAttributes::Numeric(NumericAttributes {
            unit: unit.map(ToOwned::to_owned),
            higher_is_better,
        }),
    });
    metric_ids.insert(name, id);
}

fn update_train_metric(
    renderer: &mut dyn MetricsRenderer,
    ids: &HashMap<&'static str, MetricId>,
    name: &'static str,
    value: f32,
) {
    renderer.update_train(metric_state(ids, name, value));
}

fn update_valid_metric(
    renderer: &mut dyn MetricsRenderer,
    ids: &HashMap<&'static str, MetricId>,
    name: &'static str,
    value: f32,
) {
    renderer.update_valid(metric_state(ids, name, value));
}

fn metric_state(
    ids: &HashMap<&'static str, MetricId>,
    name: &'static str,
    value: f32,
) -> MetricState {
    let formatted = format!("{value:.5}");
    let entry = MetricEntry::new(
        ids.get(name).expect("metric must be registered").clone(),
        SerializedEntry::new(formatted, value.to_string()),
    );
    MetricState::Numeric(entry, NumericEntry::Value(value as f64))
}

fn train_progress(metrics: &TrainStepMetrics, epochs: usize) -> TrainingProgress {
    train_progress_from_values(
        metrics.epoch,
        metrics.step,
        metrics.epoch_samples_processed,
        metrics.epoch_samples_total,
        epochs,
    )
}

fn valid_progress(epoch: usize, step: usize, epochs: usize) -> TrainingProgress {
    TrainingProgress {
        progress: Some(Progress {
            items_processed: epoch,
            items_total: epochs.max(1),
        }),
        global_progress: Progress {
            items_processed: epoch,
            items_total: epochs,
        },
        iteration: Some(step),
    }
}

fn train_progress_indicators(metrics: &TrainStepMetrics, epochs: usize) -> Vec<ProgressType> {
    train_progress_indicators_from_values(
        metrics.epoch,
        metrics.step,
        metrics.epoch_batch,
        metrics.epoch_batches,
        metrics.epoch_samples_processed,
        metrics.epoch_samples_total,
        epochs,
    )
}

fn train_progress_from_values(
    epoch: usize,
    step: usize,
    epoch_samples_processed: usize,
    epoch_samples_total: usize,
    epochs: usize,
) -> TrainingProgress {
    TrainingProgress {
        progress: Some(Progress {
            items_processed: epoch_samples_processed,
            items_total: epoch_samples_total.max(1),
        }),
        global_progress: Progress {
            items_processed: epoch,
            items_total: epochs,
        },
        iteration: Some(step),
    }
}

fn train_progress_indicators_from_values(
    epoch: usize,
    step: usize,
    _epoch_batch: usize,
    _epoch_batches: usize,
    _epoch_samples_processed: usize,
    _epoch_samples_total: usize,
    epochs: usize,
) -> Vec<ProgressType> {
    vec![
        ProgressType::Detailed {
            tag: "Epoch".to_string(),
            progress: Progress {
                items_processed: epoch,
                items_total: epochs,
            },
        },
        ProgressType::Value {
            tag: "Iteration".to_string(),
            value: step,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tui_metric_budget_prioritizes_validation_visibility() {
        assert_eq!(TRAIN_METRICS, ["loss", "samples/s"]);
        assert_eq!(
            VALID_METRICS,
            ["loss", "f1", "precision", "recall", "mean_iou"]
        );
    }

    #[test]
    fn train_status_keeps_within_burn_status_panel_height() {
        let indicators = train_progress_indicators_from_values(1, 2, 3, 4, 6, 8, 10);

        assert!(
            indicators.len() <= 2,
            "Burn's status panel is capped at six rows; mode plus two indicators leaves room without clipping"
        );
    }
}
