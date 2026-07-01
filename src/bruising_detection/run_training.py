import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .augment import CubeAugmenter
from .config import CNNConfig, DATA_DIR, RANDOM_STATE, REPORTS_DIR, SPLITS_DIR, TEST_SIZE, VAL_SIZE
from .data import LazyH5CropDataset, compute_band_stats, index_h5_samples, materialize_split_data, save_splits, seed_everything, split_files
from .model import ShallowCNN
from .trainer import CNNTrainer, summarize_predictions


def build_data(config):
    files = sorted(DATA_DIR.glob("*.h5"))
    source_splits = split_files(files, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE)
    splits = materialize_split_data(source_splits, SPLITS_DIR)

    train_meta = index_h5_samples(splits["train"], config.crop_size)
    val_meta = index_h5_samples(splits["val"], config.crop_size)
    test_meta = index_h5_samples(splits["test"], config.crop_size)
    meta = {"train": train_meta, "val": val_meta, "test": test_meta}
    save_splits(splits, meta, SPLITS_DIR)
    input_shape = (*train_meta.loc[0, "crop_shape"], int(train_meta.loc[0, "bands"]))

    stats_ds = LazyH5CropDataset(train_meta, config.crop_size)
    mean, std = compute_band_stats(stats_ds, batch_size=config.batch_size, num_workers=config.num_workers)

    augmenter = CubeAugmenter(config) if config.augment else None
    train_ds = LazyH5CropDataset(train_meta, config.crop_size, mean, std, augmenter)
    val_ds = LazyH5CropDataset(val_meta, config.crop_size, mean, std)
    test_ds = LazyH5CropDataset(test_meta, config.crop_size, mean, std)

    loader_kwargs = {"batch_size": config.batch_size, "num_workers": config.num_workers}
    loaders = {
        "train": DataLoader(train_ds, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_ds, **loader_kwargs),
        "test": DataLoader(test_ds, **loader_kwargs),
    }
    return loaders, meta, input_shape, splits


def main():
    seed_everything()
    config = CNNConfig()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    loaders, meta, input_shape, splits = build_data(config)

    print("Files:", {name: len(paths) for name, paths in splits.items()})
    print("Samples:", {name: len(meta[name]) for name in meta})
    print("Input shape:", input_shape)
    print("Labels:", {name: meta[name]["y"].value_counts().sort_index().to_dict() for name in meta})
    print("Augmentation:", "on" if config.augment else "off")

    model = ShallowCNN(
        in_channels=input_shape[-1],
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )
    trainer = CNNTrainer(model, config)
    trainer.log_model_graph(input_shape)
    history = trainer.fit(loaders["train"], loaders["val"])
    history.to_csv(REPORTS_DIR / "cnn_history.csv", index=False)

    checkpoint = torch.load(config.best_model_path, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from {config.best_model_path}: epoch={checkpoint['epoch']} val_acc={checkpoint['val_acc']:.4f}")

    summaries, by_time = [], []
    writer = SummaryWriter(config.log_dir) if config.log_dir else None
    try:
        for split in ["train", "val", "test"]:
            pred, true = trainer.predict(loaders[split])
            summary, time_rows = summarize_predictions(split, true, pred, meta[split])
            summaries.append(summary)
            by_time.append(time_rows)

            predictions = meta[split].copy()
            predictions["y_true"] = true
            predictions["y_pred"] = pred
            predictions["y_true_label"] = predictions["y_true"].map({0: "sound", 1: "damaged"})
            predictions["y_pred_label"] = predictions["y_pred"].map({0: "sound", 1: "damaged"})
            predictions.to_csv(REPORTS_DIR / f"cnn_{split}_predictions.csv", index=False)

            matrix = confusion_matrix(true, pred, labels=[0, 1])
            pd.DataFrame(
                matrix,
                index=["true_sound", "true_damaged"],
                columns=["pred_sound", "pred_damaged"],
            ).to_csv(REPORTS_DIR / f"cnn_{split}_confusion_matrix.csv")

            report = classification_report(true, pred, target_names=["sound", "damaged"], zero_division=0)
            report_table = classification_report(true, pred, target_names=["sound", "damaged"], zero_division=0, output_dict=True)
            (REPORTS_DIR / f"cnn_{split}_classification_report.txt").write_text(report)
            pd.DataFrame(report_table).T.to_csv(REPORTS_DIR / f"cnn_{split}_classification_report.csv")

            if writer is not None:
                for metric in ["accuracy", "precision", "recall", "f1"]:
                    writer.add_scalar(f"Final/{split}/{metric}", summary[metric], 0)
                writer.add_scalar(f"Final/{split}/true_negative", summary["true_negative"], 0)
                writer.add_scalar(f"Final/{split}/false_positive", summary["false_positive"], 0)
                writer.add_scalar(f"Final/{split}/false_negative", summary["false_negative"], 0)
                writer.add_scalar(f"Final/{split}/true_positive", summary["true_positive"], 0)

            print(f"\n{split} classification report")
            print(report)
    finally:
        if writer is not None:
            writer.close()

    pd.DataFrame(summaries).to_csv(REPORTS_DIR / "cnn_summary.csv", index=False)
    pd.concat(by_time, ignore_index=True).to_csv(REPORTS_DIR / "cnn_by_timepoint.csv", index=False)


if __name__ == "__main__":
    main()
