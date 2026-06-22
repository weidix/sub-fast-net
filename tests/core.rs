use std::{collections::HashMap, fs, path::Path};

use burn::tensor::{Tensor, activation::sigmoid};
use sub_fast_net::{
    backend::CpuAutodiffBackend,
    benchmark::benchmark_train_step_backend,
    config::{BackendKind, ModelVariant, OptimizerKind, TrainConfig},
    dataset::{
        DatasetRoot, DatasetSplit, RawLabelMaskFile, RawLabelMaskRecord, SubtitleDataset,
        apply_label_masks, load_annotations, load_root, parse_yolo_label_text,
    },
    loss::{compute_loss, compute_tensor_loss_breakdown},
    metrics::{bbox_iou, match_detection_metrics},
    model::{ModelOutput, SubFastNet},
    preprocess::{
        CoordinateSpace, ImageMeta, PixelBox, PreprocessedSample, RectanglePolygon, YoloBox,
        collate_batch_with_config, crop_padding_box_for_test, crop_padding_polygon_for_test,
        hue_jitter_rgb_for_test, preprocess_sample, random_rotate_box_for_test,
        random_rotate_polygon_for_test, random_scale_box_for_test, random_scale_polygon_for_test,
        restore_box_to_original_image, restore_box_to_output_space, scale_aligned_short,
        yolo_to_pixel,
    },
    target::{
        TargetConfig, generate_targets, generate_targets_from_polygons,
        generate_targets_with_config,
    },
    train::train_backend,
    validate::{restore_gt_boxes_to_output_space, validate_model},
};

#[test]
fn parses_yolo_labels() {
    let boxes = parse_yolo_label_text("0 0.5 0.5 0.25 0.1\n", true).unwrap();
    assert_eq!(boxes.len(), 1);
    assert_eq!(boxes[0].class_id, 0);
}

#[test]
fn mixed_precision_config_field_is_rejected() {
    let output_dir = Path::new("outputs/test_removed_mixed_precision_config");
    let _ = fs::remove_dir_all(output_dir);
    fs::create_dir_all(output_dir).unwrap();
    let config_path = output_dir.join("train.toml");
    fs::write(
        &config_path,
        r#"
experiment_name = "test_removed_mixed_precision"
output_dir = "outputs/test_removed_mixed_precision_config/run"
seed = 7
backend = "cuda"
mixed_precision = "bf16"
train_roots = ["data/generated_samples1"]
val_root = "data/validation_samples"
"#,
    )
    .unwrap();

    let err = TrainConfig::from_path(&config_path).unwrap_err();
    assert!(err.to_string().contains("mixed_precision has been removed"));
}

#[test]
fn optimizer_kind_defaults_to_adam_and_parses_sgd() {
    assert_eq!(TrainConfig::default().optimizer, OptimizerKind::Adam);
    assert_eq!(TrainConfig::default().gradient_accumulation_steps, 1);

    let output_dir = Path::new("outputs/test_optimizer_config");
    let _ = fs::remove_dir_all(output_dir);
    fs::create_dir_all(output_dir).unwrap();
    let config_path = output_dir.join("train.toml");
    fs::write(
        &config_path,
        r#"
experiment_name = "test_optimizer_config"
output_dir = "outputs/test_optimizer_config/run"
seed = 7
backend = "cpu"
optimizer = "sgd"
gradient_accumulation_steps = 4
train_roots = ["data/generated_samples1"]
val_root = "data/validation_samples"
"#,
    )
    .unwrap();

    let config = TrainConfig::from_path(&config_path).unwrap();
    assert_eq!(config.optimizer, OptimizerKind::Sgd);
    assert_eq!(config.gradient_accumulation_steps, 4);
}

#[test]
fn converts_yolo_to_pixel_box() {
    let bbox = yolo_to_pixel(
        YoloBox {
            class_id: 0,
            x_center: 0.5,
            y_center: 0.5,
            width: 0.25,
            height: 0.5,
        },
        200,
        100,
    )
    .unwrap();
    assert_eq!(
        bbox,
        PixelBox {
            x1: 75.0,
            y1: 25.0,
            x2: 125.0,
            y2: 75.0
        }
    );
}

#[test]
fn converts_box_to_rectangle_polygon() {
    let polygon = PixelBox {
        x1: 1.0,
        y1: 2.0,
        x2: 5.0,
        y2: 8.0,
    }
    .to_polygon();
    assert_eq!(polygon.points[0].x, 1.0);
    assert_eq!(polygon.points[2].y, 8.0);
}

#[test]
fn aligned_short_resize_aligns_dimensions() {
    let (w, h, scale) = scale_aligned_short(1920, 1080, 640, 32);
    assert_eq!(h % 32, 0);
    assert_eq!(w % 32, 0);
    assert!(scale > 0.0);
}

#[test]
fn random_scale_transforms_box() {
    let bbox = random_scale_box_for_test(
        PixelBox {
            x1: 10.0,
            y1: 20.0,
            x2: 30.0,
            y2: 40.0,
        },
        2.0,
        0.5,
    );
    assert_eq!(bbox.x1, 20.0);
    assert_eq!(bbox.y2, 20.0);
}

#[test]
fn random_scale_transforms_polygon_with_box() {
    let bbox = PixelBox {
        x1: 10.0,
        y1: 20.0,
        x2: 30.0,
        y2: 40.0,
    };
    let polygon = random_scale_polygon_for_test(bbox.to_polygon(), 2.0, 0.5);
    assert_eq!(
        polygon.bounding_box(),
        random_scale_box_for_test(bbox, 2.0, 0.5)
    );
}

#[test]
fn random_rotate_keeps_box_valid() {
    let bbox = random_rotate_box_for_test(
        PixelBox {
            x1: 10.0,
            y1: 20.0,
            x2: 30.0,
            y2: 40.0,
        },
        5.0,
        100,
        100,
    );
    assert!(bbox.is_valid());
}

#[test]
fn random_rotate_regenerates_box_from_polygon() {
    let bbox = PixelBox {
        x1: 10.0,
        y1: 20.0,
        x2: 30.0,
        y2: 40.0,
    };
    let polygon = random_rotate_polygon_for_test(bbox.to_polygon(), 5.0, 100, 100);
    assert_eq!(
        polygon.bounding_box(),
        random_rotate_box_for_test(bbox, 5.0, 100, 100)
    );
}

#[test]
fn crop_padding_clips_box() {
    let bbox = crop_padding_box_for_test(
        PixelBox {
            x1: 10.0,
            y1: 10.0,
            x2: 30.0,
            y2: 30.0,
        },
        20,
        20,
        32,
        32,
    )
    .unwrap();
    assert_eq!(bbox.x1, 0.0);
    assert_eq!(bbox.y1, 0.0);
    assert_eq!(bbox.x2, 10.0);
    assert_eq!(bbox.y2, 10.0);
}

#[test]
fn crop_padding_clips_polygon_and_drops_outside_polygon() {
    let bbox = PixelBox {
        x1: 10.0,
        y1: 10.0,
        x2: 30.0,
        y2: 30.0,
    };
    let polygon = crop_padding_polygon_for_test(bbox.to_polygon(), 20, 20, 32, 32).unwrap();
    assert_eq!(
        polygon.bounding_box(),
        crop_padding_box_for_test(bbox, 20, 20, 32, 32).unwrap()
    );

    let outside = crop_padding_polygon_for_test(bbox.to_polygon(), 40, 40, 16, 16);
    assert!(outside.is_none());
}

#[test]
fn hue_jitter_changes_hue_without_grayscale_shift() {
    let shifted = hue_jitter_rgb_for_test([255, 0, 0], 1.0 / 3.0);
    assert!(shifted[1] > 200);
    assert!(shifted[0] < 10);

    let gray = hue_jitter_rgb_for_test([128, 128, 128], 0.2);
    assert_eq!(gray, [128, 128, 128]);
}

#[test]
fn generates_fast_style_targets() {
    let targets = generate_targets(
        16,
        16,
        &[PixelBox {
            x1: 4.0,
            y1: 4.0,
            x2: 12.0,
            y2: 12.0,
        }],
        &[PixelBox {
            x1: 0.0,
            y1: 0.0,
            x2: 2.0,
            y2: 2.0,
        }],
    );
    assert!(targets.gt_text.iter().any(|value| *value > 0.0));
    assert!(targets.gt_kernel.iter().any(|value| *value > 0.0));
    assert_eq!(targets.training_mask[0], 0.0);
    assert!(targets.gt_instance.contains(&1));
}

#[test]
fn generates_targets_from_rectangle_polygon_semantics() {
    let polygon = PixelBox {
        x1: 4.0,
        y1: 4.0,
        x2: 12.0,
        y2: 12.0,
    }
    .to_polygon();
    let targets = generate_targets_from_polygons(
        16,
        16,
        &[polygon],
        &[PixelBox {
            x1: 0.0,
            y1: 0.0,
            x2: 2.0,
            y2: 2.0,
        }],
    );
    assert!(targets.gt_instance.contains(&1));
    assert!(targets.gt_text.iter().any(|value| *value > 0.0));
    assert!(targets.gt_kernel.iter().any(|value| *value > 0.0));
    assert_eq!(targets.training_mask[0], 0.0);
}

#[test]
fn parameterized_kernel_preserves_small_subtitles() {
    let targets = generate_targets_with_config(
        32,
        16,
        &[PixelBox {
            x1: 4.0,
            y1: 6.0,
            x2: 28.0,
            y2: 9.0,
        }],
        &[],
        TargetConfig {
            pooling_size: 9,
            shrink_kernel_scale: 0.1,
            min_kernel_width: 3,
            min_kernel_height: 3,
        },
    );
    assert!(targets.gt_kernel.iter().any(|value| *value > 0.0));
}

#[test]
fn applies_label_mask_full_semantics() {
    let mut sample_masks = HashMap::new();
    sample_masks.insert(
        "0".to_string(),
        RawLabelMaskRecord {
            masked: Some(true),
            ..RawLabelMaskRecord::default()
        },
    );
    sample_masks.insert(
        "1".to_string(),
        RawLabelMaskRecord {
            bbox: Some([20.0, 20.0, 40.0, 40.0]),
            unreliable: Some(true),
            ..RawLabelMaskRecord::default()
        },
    );
    sample_masks.insert(
        "add".to_string(),
        RawLabelMaskRecord {
            add_bbox: Some([50.0, 50.0, 70.0, 70.0]),
            ignore_region: Some([0.0, 0.0, 5.0, 5.0]),
            ..RawLabelMaskRecord::default()
        },
    );
    let masks = RawLabelMaskFile {
        items: HashMap::from([("sample".to_string(), sample_masks)]),
        ..RawLabelMaskFile::default()
    };
    let (boxes, result) = apply_label_masks(
        &masks,
        "sample",
        &[
            PixelBox {
                x1: 1.0,
                y1: 1.0,
                x2: 10.0,
                y2: 10.0,
            },
            PixelBox {
                x1: 10.0,
                y1: 10.0,
                x2: 30.0,
                y2: 30.0,
            },
        ],
        100,
        100,
        true,
    )
    .unwrap();
    assert_eq!(result.deleted_count, 1);
    assert_eq!(result.corrected_count, 1);
    assert_eq!(result.added_count, 1);
    assert_eq!(result.unreliable_count, 1);
    assert_eq!(boxes.len(), 2);
    assert_eq!(result.ignore_regions.len(), 2);
}

#[test]
fn reads_annotations_metadata() {
    let annotations =
        load_annotations(Path::new("data/validation_samples/annotations.jsonl"), true).unwrap();
    let meta = annotations.get("video0001_f00000060").unwrap();
    assert_eq!(meta.image_width, Some(1920));
    assert_eq!(meta.image_height, Some(1080));
    assert_eq!(meta.frame_index, Some(60));
    assert!(meta.source_video.is_some());
}

#[test]
fn merges_multiple_dataset_roots() {
    let roots = vec![
        load_root(0, Path::new("data/generated_samples1"), false).unwrap(),
        load_root(1, Path::new("data/generated_samples2"), false).unwrap(),
    ];
    let dataset = SubtitleDataset::from_roots(DatasetSplit::Train, false, roots).unwrap();
    assert!(dataset.len() > 2);
    assert_eq!(dataset.sample_indices()[0].root_id, 0);
    assert!(
        dataset
            .sample_indices()
            .iter()
            .any(|sample| sample.root_id == 1)
    );
}

#[test]
fn max_train_samples_is_balanced_across_roots() {
    let mut config = smoke_test_config("outputs/test_balanced_samples");
    config.train_roots = vec![
        "data/generated_samples1".to_string(),
        "data/generated_samples2".to_string(),
        "data/generated_samples3".to_string(),
    ];
    config.max_train_samples = Some(3);
    let dataset = SubtitleDataset::from_train_config(&config).unwrap();
    let roots = dataset
        .sample_indices()
        .iter()
        .map(|sample| sample.root_id)
        .collect::<Vec<_>>();
    assert_eq!(roots.len(), 3);
    assert!(roots.contains(&0));
    assert!(roots.contains(&1));
    assert!(roots.contains(&2));
}

#[test]
fn max_train_samples_keeps_labeled_training_examples() {
    let mut config = smoke_test_config("outputs/test_labeled_sample_limit");
    config.train_roots = vec!["data/generated_samples1".to_string()];
    config.max_train_samples = Some(2);
    let dataset = SubtitleDataset::from_train_config(&config).unwrap();

    let labeled_count = (0..dataset.len())
        .map(|index| dataset.load_sample(index).unwrap())
        .filter(|sample| !sample.pixel_boxes_after_label_masks.is_empty())
        .count();

    assert!(
        labeled_count > 0,
        "limited training dataset must include at least one labeled subtitle sample"
    );
}

#[test]
fn max_train_samples_can_reserve_empty_training_examples() {
    let mut config = smoke_test_config("outputs/test_empty_sample_reserve");
    config.train_roots = vec!["data/generated_samples1".to_string()];
    config.max_train_samples = Some(20);
    config.train_empty_sample_ratio = Some(0.25);
    let dataset = SubtitleDataset::from_train_config(&config).unwrap();

    let empty_count = (0..dataset.len())
        .map(|index| dataset.load_sample(index).unwrap())
        .filter(|sample| sample.pixel_boxes_after_label_masks.is_empty())
        .count();

    assert!(
        empty_count >= 5,
        "limited training dataset must reserve configured empty subtitle samples"
    );
}

#[test]
fn train_empty_sample_ratio_does_not_create_empty_only_batches() {
    let mut config = smoke_test_config("outputs/test_empty_sample_interleave");
    config.train_roots = vec!["data/generated_samples1".to_string()];
    config.max_train_samples = Some(20);
    config.train_empty_sample_ratio = Some(0.25);
    config.batch_size = 2;
    let dataset = SubtitleDataset::from_train_config(&config).unwrap();

    for batch_start in (0..dataset.len()).step_by(config.batch_size) {
        let batch_end = (batch_start + config.batch_size).min(dataset.len());
        let labeled_count = (batch_start..batch_end)
            .map(|index| dataset.load_sample(index).unwrap())
            .filter(|sample| !sample.pixel_boxes_after_label_masks.is_empty())
            .count();
        assert!(
            labeled_count > 0,
            "empty sample ratio must be interleaved, not concentrated into empty-only batches"
        );
    }
}

#[test]
fn max_train_samples_are_spread_across_each_root() {
    let mut full_config = smoke_test_config("outputs/test_spread_full");
    full_config.train_roots = vec!["data/generated_samples1".to_string()];
    full_config.max_train_samples = None;
    let full_dataset = SubtitleDataset::from_train_config(&full_config).unwrap();
    let first_twenty_labeled = (0..full_dataset.len())
        .filter_map(|index| {
            let sample = full_dataset.load_sample(index).unwrap();
            (!sample.pixel_boxes_after_label_masks.is_empty()).then_some(sample.sample_id)
        })
        .take(20)
        .collect::<Vec<_>>();

    let mut limited_config = full_config;
    limited_config.max_train_samples = Some(20);
    let limited_dataset = SubtitleDataset::from_train_config(&limited_config).unwrap();
    let limited = limited_dataset
        .sample_indices()
        .iter()
        .map(|sample| sample.sample_id.clone())
        .collect::<Vec<_>>();

    assert_ne!(
        limited, first_twenty_labeled,
        "limited training samples should be spread through each root, not only the sorted prefix"
    );
}

#[test]
fn max_val_samples_keeps_labeled_validation_examples() {
    let mut config = smoke_test_config("outputs/test_labeled_val_limit");
    config.max_val_samples = Some(2);
    let dataset = SubtitleDataset::from_val_config(&config).unwrap();

    let labeled_count = (0..dataset.len())
        .map(|index| dataset.load_sample(index).unwrap())
        .filter(|sample| !sample.pixel_boxes_after_label_masks.is_empty())
        .count();

    assert!(
        labeled_count > 0,
        "limited validation dataset must include at least one labeled subtitle sample"
    );
}

#[test]
fn handles_empty_labels_and_strict_missing_root() {
    let dataset = SubtitleDataset::from_roots(
        DatasetSplit::Train,
        false,
        vec![DatasetRoot {
            root_id: 0,
            path: Path::new("data/generated_samples1").to_path_buf(),
            annotations: HashMap::new(),
            label_masks: RawLabelMaskFile::default(),
        }],
    )
    .unwrap();
    let report = dataset.inspect();
    assert!(report.empty_label_count > 0);
    assert!(load_root(0, Path::new("missing-root-for-test"), true).is_err());
}

#[test]
fn preprocess_collate_and_forward_smoke() {
    let mut config = smoke_test_config("outputs/test_forward_smoke");
    config.max_train_samples = Some(1);
    let dataset = SubtitleDataset::from_train_config(&config).unwrap();
    let sample = dataset.load_sample(0).unwrap();
    let preprocessed = preprocess_sample(&sample, &config, true).unwrap();
    let batch = collate_batch_with_config(vec![preprocessed], &config);
    assert_eq!(batch.imgs.len(), 1);
    assert_eq!(batch.gt_texts.len(), 1);
    let device = Default::default();
    let model = SubFastNet::<CpuAutodiffBackend>::new(ModelVariant::Tiny, &device);
    let output = model.forward(batch.image_tensor::<CpuAutodiffBackend>(&device));
    assert_eq!(output.text_region_logits.dims()[1], 1);
    assert_eq!(output.kernel_logits.dims()[1], 1);
    assert_eq!(output.height, batch.height / 4);
    assert_eq!(output.width, batch.width / 4);
    let arch = model.architecture();
    assert_eq!(arch.feature_strides, [4, 8, 16]);
    assert_eq!(arch.output_stride, 4);
    assert_eq!(arch.detection_head_count, 2);
}

#[test]
fn disabled_augmentation_uses_validation_geometry_for_training() {
    let mut config = smoke_test_config("outputs/test_no_aug_training_geometry");
    config.augment_enabled = false;
    config.max_train_samples = Some(2);
    let dataset = SubtitleDataset::from_train_config(&config).unwrap();
    let sample = (0..dataset.len())
        .map(|index| dataset.load_sample(index).unwrap())
        .find(|sample| !sample.pixel_boxes_after_label_masks.is_empty())
        .unwrap();

    let training = preprocess_sample(&sample, &config, true).unwrap();
    let validation = preprocess_sample(&sample, &config, false).unwrap();

    assert_eq!(training.width, validation.width);
    assert_eq!(training.height, validation.height);
    assert_eq!(training.boxes, validation.boxes);
}

#[test]
fn model_size_estimate_stays_in_target_range() {
    let bytes = sub_fast_net::model::serialized_size_bytes_estimate(ModelVariant::Tiny);
    assert!((1_000_000..=4_000_000).contains(&bytes));
}

#[test]
fn smoke_train_saves_checkpoint_optimizer_scheduler() {
    let output_dir = "outputs/test_smoke_train";
    let _ = fs::remove_dir_all(output_dir);
    let config = smoke_test_config(output_dir);
    let summary = train_backend::<CpuAutodiffBackend>(&config).unwrap();
    assert_eq!(summary.final_epoch, 1);
    assert!(Path::new(output_dir).join("best/model.bin").is_file());
    assert!(Path::new(output_dir).join("best/optimizer.bin").is_file());
    assert!(Path::new(output_dir).join("best/scheduler.json").is_file());
    assert!(
        Path::new(output_dir)
            .join("errors/false_positive.jsonl")
            .is_file()
    );
}

#[test]
fn collate_batch_pads_variable_size_samples_to_common_shape() {
    let config = smoke_test_config("outputs/test_variable_size_collate");
    let wide = preprocessed_sample_for_collate("wide", 4, 2, 1.0);
    let tall = preprocessed_sample_for_collate("tall", 2, 3, 2.0);

    let batch = collate_batch_with_config(vec![wide, tall], &config);

    assert_eq!(batch.width, 4);
    assert_eq!(batch.height, 3);
    assert_eq!(batch.imgs[0].len(), 3 * 4 * 3);
    assert_eq!(batch.imgs[1].len(), 3 * 4 * 3);
}

#[test]
fn validation_summary_reports_postprocess_latency() {
    let mut config = smoke_test_config("outputs/test_validate_postprocess_latency");
    config.max_val_samples = Some(1);
    let val_dataset = SubtitleDataset::from_val_config(&config).unwrap();
    let device = Default::default();
    let model = SubFastNet::<sub_fast_net::backend::CpuBackend>::new(ModelVariant::Tiny, &device);
    let summary = validate_model(&config, &val_dataset, &model).unwrap();
    assert!(summary.postprocess_latency >= 0.0);
}

#[test]
fn restores_postprocess_box_to_original_pixels() {
    let meta = ImageMeta {
        image_path: "image.jpg".to_string(),
        sample_id: "image".to_string(),
        original_width: 100,
        original_height: 50,
        resized_width: 200,
        resized_height: 100,
        scale: 2.0,
        pad: [0, 0, 0, 0],
        source: Some("video.mp4".to_string()),
        frame_id: Some("42".to_string()),
        coordinate_space: CoordinateSpace::Image,
        roi_offset: None,
        frame_width: None,
        frame_height: None,
    };
    let restored = restore_box_to_original_image(
        PixelBox {
            x1: 20.0,
            y1: 10.0,
            x2: 60.0,
            y2: 30.0,
        },
        &meta,
    );
    assert_eq!(restored.x1, 10.0);
    assert_eq!(restored.y2, 15.0);
}

#[test]
fn restores_roi_crop_box_to_original_frame_pixels() {
    let meta = ImageMeta {
        image_path: "roi.jpg".to_string(),
        sample_id: "roi".to_string(),
        original_width: 320,
        original_height: 120,
        resized_width: 640,
        resized_height: 240,
        scale: 2.0,
        pad: [0, 0, 0, 0],
        source: Some("video.mp4".to_string()),
        frame_id: Some("42".to_string()),
        coordinate_space: CoordinateSpace::OriginalFrame,
        roi_offset: Some([100, 700]),
        frame_width: Some(1920),
        frame_height: Some(1080),
    };
    let restored = restore_box_to_output_space(
        PixelBox {
            x1: 20.0,
            y1: 10.0,
            x2: 60.0,
            y2: 30.0,
        },
        &meta,
    );
    assert_eq!(restored.x1, 110.0);
    assert_eq!(restored.y1, 705.0);
    assert_eq!(restored.x2, 130.0);
    assert_eq!(restored.y2, 715.0);
}

#[test]
fn validation_restores_gt_boxes_to_output_space() {
    let meta = ImageMeta {
        image_path: "image.jpg".to_string(),
        sample_id: "image".to_string(),
        original_width: 100,
        original_height: 50,
        resized_width: 200,
        resized_height: 100,
        scale: 2.0,
        pad: [0, 0, 0, 0],
        source: Some("video.mp4".to_string()),
        frame_id: Some("42".to_string()),
        coordinate_space: CoordinateSpace::Image,
        roi_offset: None,
        frame_width: None,
        frame_height: None,
    };

    let restored = restore_gt_boxes_to_output_space(
        &[PixelBox {
            x1: 20.0,
            y1: 10.0,
            x2: 60.0,
            y2: 30.0,
        }],
        &meta,
    );

    assert_eq!(
        restored,
        vec![PixelBox {
            x1: 10.0,
            y1: 5.0,
            x2: 30.0,
            y2: 15.0,
        }]
    );
}

#[test]
fn benchmark_measures_real_train_step_time() {
    let mut config = smoke_test_config("outputs/test_benchmark_train_step");
    config.augment_enabled = false;
    let ms = benchmark_train_step_backend::<CpuAutodiffBackend>(&config).unwrap();
    assert!(ms > 0.0);
}

#[test]
fn tensor_loss_breakdown_detaches_logging_fields() {
    let device = Default::default();
    let region_logits =
        Tensor::<CpuAutodiffBackend, 1>::from_floats([0.25, -0.5, 0.75, -1.0], &device)
            .reshape([1, 1, 2, 2])
            .require_grad();
    let kernel_logits =
        Tensor::<CpuAutodiffBackend, 1>::from_floats([-0.25, 0.5, -0.75, 1.0], &device)
            .reshape([1, 1, 2, 2])
            .require_grad();
    let output = ModelOutput {
        text_region_logits: region_logits,
        kernel_logits,
        width: 2,
        height: 2,
    };
    let gt_text = Tensor::<CpuAutodiffBackend, 1>::from_floats([1.0, 0.0, 1.0, 0.0], &device)
        .reshape([1, 1, 2, 2]);
    let gt_kernel = Tensor::<CpuAutodiffBackend, 1>::from_floats([0.0, 1.0, 0.0, 1.0], &device)
        .reshape([1, 1, 2, 2]);
    let training_mask = Tensor::<CpuAutodiffBackend, 1>::from_floats([1.0, 1.0, 1.0, 0.0], &device)
        .reshape([1, 1, 2, 2]);

    let loss = compute_tensor_loss_breakdown(&output, gt_text, gt_kernel, training_mask);

    let _ = loss.total_loss.clone().backward();
    assert!(!loss.region_bce_loss.is_require_grad());
    assert!(!loss.region_dice_loss.is_require_grad());
    assert!(!loss.kernel_bce_loss.is_require_grad());
    assert!(!loss.kernel_dice_loss.is_require_grad());
    assert!(!loss.ignored_area_ratio.is_require_grad());
    assert!(!loss.positive_region_ratio.is_require_grad());
    assert!(!loss.positive_kernel_ratio.is_require_grad());
}

#[test]
fn tensor_loss_breakdown_matches_cpu_reference() {
    let device = Default::default();
    let region_values = vec![0.25, -0.5, 0.75, -1.0, 0.1, 1.2, -1.4, 0.0];
    let kernel_values = vec![-0.25, 0.5, -0.75, 1.0, -0.2, 0.8, -0.9, 0.3];
    let gt_text_values = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let gt_kernel_values = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let mask_values = vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];

    let output = ModelOutput {
        text_region_logits: Tensor::<CpuAutodiffBackend, 1>::from_floats(
            region_values.as_slice(),
            &device,
        )
        .reshape([2, 1, 2, 2])
        .require_grad(),
        kernel_logits: Tensor::<CpuAutodiffBackend, 1>::from_floats(
            kernel_values.as_slice(),
            &device,
        )
        .reshape([2, 1, 2, 2])
        .require_grad(),
        width: 2,
        height: 2,
    };
    let gt_text = Tensor::<CpuAutodiffBackend, 1>::from_floats(gt_text_values.as_slice(), &device)
        .reshape([2, 1, 2, 2]);
    let gt_kernel =
        Tensor::<CpuAutodiffBackend, 1>::from_floats(gt_kernel_values.as_slice(), &device)
            .reshape([2, 1, 2, 2]);
    let training_mask =
        Tensor::<CpuAutodiffBackend, 1>::from_floats(mask_values.as_slice(), &device)
            .reshape([2, 1, 2, 2]);

    let tensor_loss = compute_tensor_loss_breakdown(&output, gt_text, gt_kernel, training_mask)
        .to_cpu_loss_breakdown();
    let reference = compute_loss(
        &sub_fast_net::model::output_to_cpu(output),
        &[gt_text_values[..4].to_vec(), gt_text_values[4..].to_vec()],
        &[
            gt_kernel_values[..4].to_vec(),
            gt_kernel_values[4..].to_vec(),
        ],
        &[mask_values[..4].to_vec(), mask_values[4..].to_vec()],
    );
    let max_error = [
        (tensor_loss.total_loss - reference.total_loss).abs(),
        (tensor_loss.region_bce_loss - reference.region_bce_loss).abs(),
        (tensor_loss.region_dice_loss - reference.region_dice_loss).abs(),
        (tensor_loss.kernel_bce_loss - reference.kernel_bce_loss).abs(),
        (tensor_loss.kernel_dice_loss - reference.kernel_dice_loss).abs(),
        (tensor_loss.ignored_area_ratio - reference.ignored_area_ratio).abs(),
        (tensor_loss.positive_region_ratio - reference.positive_region_ratio).abs(),
        (tensor_loss.positive_kernel_ratio - reference.positive_kernel_ratio).abs(),
    ]
    .into_iter()
    .fold(0.0_f32, f32::max);

    assert!(max_error <= 1e-6, "max_error={max_error}");
}

#[test]
fn tensor_loss_breakdown_matches_legacy_tensor_formula() {
    let device = Default::default();
    let region_values = vec![0.25, -0.5, 0.75, -1.0, 0.1, 1.2, -1.4, 0.0];
    let kernel_values = vec![-0.25, 0.5, -0.75, 1.0, -0.2, 0.8, -0.9, 0.3];
    let gt_text_values = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let gt_kernel_values = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let mask_values = vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];

    let output = ModelOutput {
        text_region_logits: Tensor::<CpuAutodiffBackend, 1>::from_floats(
            region_values.as_slice(),
            &device,
        )
        .reshape([2, 1, 2, 2])
        .require_grad(),
        kernel_logits: Tensor::<CpuAutodiffBackend, 1>::from_floats(
            kernel_values.as_slice(),
            &device,
        )
        .reshape([2, 1, 2, 2])
        .require_grad(),
        width: 2,
        height: 2,
    };
    let gt_text = Tensor::<CpuAutodiffBackend, 1>::from_floats(gt_text_values.as_slice(), &device)
        .reshape([2, 1, 2, 2]);
    let gt_kernel =
        Tensor::<CpuAutodiffBackend, 1>::from_floats(gt_kernel_values.as_slice(), &device)
            .reshape([2, 1, 2, 2]);
    let training_mask =
        Tensor::<CpuAutodiffBackend, 1>::from_floats(mask_values.as_slice(), &device)
            .reshape([2, 1, 2, 2]);

    let optimized = compute_tensor_loss_breakdown(
        &output,
        gt_text.clone(),
        gt_kernel.clone(),
        training_mask.clone(),
    )
    .to_cpu_loss_breakdown();
    let legacy = legacy_tensor_loss_breakdown_values(&output, gt_text, gt_kernel, training_mask);
    let max_error = [
        (optimized.total_loss - legacy.total_loss).abs(),
        (optimized.region_bce_loss - legacy.region_bce_loss).abs(),
        (optimized.region_dice_loss - legacy.region_dice_loss).abs(),
        (optimized.kernel_bce_loss - legacy.kernel_bce_loss).abs(),
        (optimized.kernel_dice_loss - legacy.kernel_dice_loss).abs(),
        (optimized.ignored_area_ratio - legacy.ignored_area_ratio).abs(),
        (optimized.positive_region_ratio - legacy.positive_region_ratio).abs(),
        (optimized.positive_kernel_ratio - legacy.positive_kernel_ratio).abs(),
    ]
    .into_iter()
    .fold(0.0_f32, f32::max);

    assert!(max_error <= 1e-6, "max_error={max_error}");
}

#[test]
fn tensor_bce_uses_logit_stable_unsaturated_formula() {
    let device = Default::default();
    let output = ModelOutput {
        text_region_logits: Tensor::<CpuAutodiffBackend, 1>::from_floats([-20.0, 20.0], &device)
            .reshape([1, 1, 1, 2])
            .require_grad(),
        kernel_logits: Tensor::<CpuAutodiffBackend, 1>::from_floats([20.0, -20.0], &device)
            .reshape([1, 1, 1, 2])
            .require_grad(),
        width: 2,
        height: 1,
    };
    let gt_text =
        Tensor::<CpuAutodiffBackend, 1>::from_floats([1.0, 0.0], &device).reshape([1, 1, 1, 2]);
    let gt_kernel =
        Tensor::<CpuAutodiffBackend, 1>::from_floats([0.0, 1.0], &device).reshape([1, 1, 1, 2]);
    let training_mask =
        Tensor::<CpuAutodiffBackend, 1>::from_floats([1.0, 1.0], &device).reshape([1, 1, 1, 2]);

    let loss = compute_tensor_loss_breakdown(&output, gt_text, gt_kernel, training_mask)
        .to_cpu_loss_breakdown();

    assert!(
        (loss.region_bce_loss - 20.0).abs() <= 1e-4,
        "region_bce_loss={}",
        loss.region_bce_loss
    );
    assert!(
        (loss.kernel_bce_loss - 20.0).abs() <= 1e-4,
        "kernel_bce_loss={}",
        loss.kernel_bce_loss
    );
}

#[test]
fn sparse_positive_bce_is_balanced_against_background() {
    let output = sub_fast_net::model::CpuModelOutput {
        text_region_logits: vec![vec![0.0, 0.0, 0.0, 0.0]],
        kernel_logits: vec![vec![0.0, 0.0, 0.0, 0.0]],
        width: 4,
        height: 1,
    };
    let loss = compute_loss(
        &output,
        &[vec![1.0, 0.0, 0.0, 0.0]],
        &[vec![1.0, 0.0, 0.0, 0.0]],
        &[vec![1.0, 1.0, 1.0, 1.0]],
    );

    assert!(
        (loss.region_bce_loss - std::f32::consts::LN_2 * 1.5).abs() <= 1e-6,
        "region_bce_loss={}",
        loss.region_bce_loss
    );

    let missing_positive = sub_fast_net::model::CpuModelOutput {
        text_region_logits: vec![vec![-2.0, 0.0, 0.0, 0.0]],
        kernel_logits: vec![vec![-2.0, 0.0, 0.0, 0.0]],
        width: 4,
        height: 1,
    };
    let missing_loss = compute_loss(
        &missing_positive,
        &[vec![1.0, 0.0, 0.0, 0.0]],
        &[vec![1.0, 0.0, 0.0, 0.0]],
        &[vec![1.0, 1.0, 1.0, 1.0]],
    );

    assert!(
        missing_loss.region_bce_loss - loss.region_bce_loss > 0.5,
        "missing positive should not be diluted by three background pixels"
    );
}

fn legacy_tensor_loss_breakdown_values(
    output: &ModelOutput<CpuAutodiffBackend>,
    gt_text: Tensor<CpuAutodiffBackend, 4>,
    gt_kernel: Tensor<CpuAutodiffBackend, 4>,
    training_mask: Tensor<CpuAutodiffBackend, 4>,
) -> sub_fast_net::loss::LossBreakdown {
    let region_bce = legacy_masked_bce_tensor(
        output.text_region_logits.clone(),
        gt_text.clone(),
        training_mask.clone(),
    );
    let kernel_bce = legacy_masked_bce_tensor(
        output.kernel_logits.clone(),
        gt_kernel.clone(),
        training_mask.clone(),
    );
    let region_dice = legacy_masked_dice_tensor(
        output.text_region_logits.clone(),
        gt_text.clone(),
        training_mask.clone(),
    );
    let kernel_dice = legacy_masked_dice_tensor(
        output.kernel_logits.clone(),
        gt_kernel.clone(),
        training_mask.clone(),
    );
    let pixels = training_mask.clone().zeros_like() + 1.0;
    let pixel_count = pixels.sum().clamp_min(1.0);
    let valid_count = training_mask.clone().sum().clamp_min(1.0);
    let ignored_count = pixel_count.clone() - training_mask.clone().sum();
    let total_loss =
        region_bce.clone() + kernel_bce.clone() + region_dice.clone() + kernel_dice.clone();
    sub_fast_net::loss::LossBreakdown {
        total_loss: scalar_tensor_value(total_loss),
        region_bce_loss: scalar_tensor_value(region_bce),
        region_dice_loss: scalar_tensor_value(region_dice),
        kernel_bce_loss: scalar_tensor_value(kernel_bce),
        kernel_dice_loss: scalar_tensor_value(kernel_dice),
        bbox_loss: 0.0,
        ignored_area_ratio: scalar_tensor_value(ignored_count / pixel_count),
        positive_region_ratio: scalar_tensor_value(
            (gt_text * training_mask.clone()).sum() / valid_count.clone(),
        ),
        positive_kernel_ratio: scalar_tensor_value((gt_kernel * training_mask).sum() / valid_count),
    }
}

fn legacy_masked_bce_tensor(
    logits: Tensor<CpuAutodiffBackend, 4>,
    targets: Tensor<CpuAutodiffBackend, 4>,
    mask: Tensor<CpuAutodiffBackend, 4>,
) -> Tensor<CpuAutodiffBackend, 1> {
    let probs = sigmoid(logits).clamp(1e-6, 1.0 - 1e-6);
    let one = targets.clone().zeros_like() + 1.0;
    let bce = (targets.clone() * probs.clone().log()
        + (one.clone() - targets) * (one - probs).log())
    .neg();
    let masked = bce * mask.clone();
    masked.sum() / mask.sum().clamp_min(1.0)
}

fn legacy_masked_dice_tensor(
    logits: Tensor<CpuAutodiffBackend, 4>,
    targets: Tensor<CpuAutodiffBackend, 4>,
    mask: Tensor<CpuAutodiffBackend, 4>,
) -> Tensor<CpuAutodiffBackend, 1> {
    let probs = sigmoid(logits) * mask.clone();
    let targets = targets * mask;
    let intersection = (probs.clone() * targets.clone()).sum();
    let denom = probs.sum() + targets.sum();
    1.0 - ((intersection * 2.0 + 1.0) / (denom + 1.0))
}

fn scalar_tensor_value(tensor: Tensor<CpuAutodiffBackend, 1>) -> f32 {
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .expect("scalar tensor should be f32");
    values[0]
}

#[test]
fn benchmark_writes_design_standard_outputs() {
    let output_dir = "outputs/test_benchmark_outputs";
    let _ = fs::remove_dir_all(output_dir);
    let config = smoke_test_config(output_dir);
    train_backend::<CpuAutodiffBackend>(&config).unwrap();
    let summary = sub_fast_net::benchmark::benchmark_backend::<sub_fast_net::backend::CpuBackend>(
        &config,
        &format!("{output_dir}/best"),
    )
    .unwrap();
    assert_eq!(summary.record_type, "benchmark");
    let summary_text = fs::read_to_string(Path::new(output_dir).join("summary.json")).unwrap();
    assert!(summary_text.contains("\"benchmark\""));
    let metrics_text = fs::read_to_string(Path::new(output_dir).join("metrics.jsonl")).unwrap();
    assert!(metrics_text.contains("\"record_type\":\"benchmark\""));
    assert!(
        Path::new(output_dir)
            .join("benchmark_summary.json")
            .is_file()
    );
    assert!(
        Path::new(output_dir)
            .join("benchmark_metrics.jsonl")
            .is_file()
    );
}

#[test]
fn computes_iou_metrics() {
    let a = PixelBox {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
    };
    let b = PixelBox {
        x1: 5.0,
        y1: 5.0,
        x2: 15.0,
        y2: 15.0,
    };
    assert!(bbox_iou(a, b) > 0.0);
    let metrics = match_detection_metrics(&[a], &[a], 0.5);
    assert_eq!(metrics.false_negative_count, 0);
    assert_eq!(metrics.false_positive_count, 0);
}

fn preprocessed_sample_for_collate(
    sample_id: &str,
    width: usize,
    height: usize,
    value: f32,
) -> PreprocessedSample {
    PreprocessedSample {
        image: vec![value; 3 * width * height],
        channels: 3,
        width,
        height,
        boxes: Vec::new(),
        rectangle_polygons: Vec::<RectanglePolygon>::new(),
        ignore_regions: Vec::new(),
        meta: ImageMeta {
            image_path: format!("{sample_id}.jpg"),
            sample_id: sample_id.to_string(),
            original_width: width as u32,
            original_height: height as u32,
            resized_width: width as u32,
            resized_height: height as u32,
            scale: 1.0,
            pad: [0, 0, 0, 0],
            source: None,
            frame_id: None,
            coordinate_space: CoordinateSpace::Image,
            roi_offset: None,
            frame_width: None,
            frame_height: None,
        },
    }
}

fn smoke_test_config(output_dir: &str) -> TrainConfig {
    TrainConfig {
        experiment_name: "test_smoke".to_string(),
        output_dir: output_dir.to_string(),
        seed: 7,
        backend: BackendKind::Cpu,
        train_roots: vec!["data/generated_samples1".to_string()],
        val_root: "data/validation_samples".to_string(),
        model_variant: ModelVariant::Tiny,
        input_size: 64,
        short_size: 64,
        alignment: 32,
        batch_size: 1,
        epochs: 1,
        learning_rate: 0.001,
        validation_interval: 1,
        checkpoint_interval: 1,
        log_interval: 1,
        threshold_region: 0.5,
        threshold_kernel: 0.5,
        iou_threshold: 0.5,
        pooling_size: 9,
        shrink_kernel_scale: 0.1,
        min_kernel_width: 3,
        min_kernel_height: 3,
        augment_enabled: true,
        strict_dataset: false,
        tui_enabled: false,
        resume: String::new(),
        max_train_samples: Some(1),
        max_val_samples: Some(1),
        ..TrainConfig::default()
    }
}
