from dataclasses import replace
from itertools import product

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .augment import CubeAugmenter
from .config import CNNConfig, DATA_DIR, PROJECT_ROOT, REPORTS_DIR, SPLITS_DIR, TEST_SIZE, VAL_SIZE
from .data import LazyH5CropDataset, compute_band_stats, index_h5_samples, materialize_split_data, save_splits, seed_everything, split_files
from .model import ShallowCNN
from .trainer import CNNTrainer, summarize_predictions


CNN_SPLIT_RANDOM_STATE = 40
RUN_NAME = "cnn_gridsearch"
RUN_REPORTS_DIR = REPORTS_DIR
RUN_TENSORBOARD_DIR = PROJECT_ROOT / "reports" / "tensorboard"
RUN_MODELS_DIR = PROJECT_ROOT / "models"
RUN_SPLITS_DIR = SPLITS_DIR

CNN_GRID = {
    "lr": [1e-3, 3e-4],
    "dropout": [0.20, 0.35],
    "weight_decay": [1e-4],
    "conv_channels": [(16, 32)],
}


def grid_candidates():
    keys = list(CNN_GRID)
    for values in product(*(CNN_GRID[key] for key in keys)):
        yield dict(zip(keys, values))


def trial_name(index, params):
    channels = "-".join(str(v) for v in params["conv_channels"])
    lr = str(params["lr"]).replace(".", "p")
    dropout = str(params["dropout"]).replace(".", "p")
    return f"trial{index:02d}_lr{lr}_dropout{dropout}_ch{channels}"


def build_data(config, split_seed=CNN_SPLIT_RANDOM_STATE, splits_dir=RUN_SPLITS_DIR):
    files = sorted(DATA_DIR.glob("*.h5"))
    source_splits = split_files(files, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=split_seed)
    splits = materialize_split_data(source_splits, splits_dir)

    train_meta = index_h5_samples(splits["train"], config.crop_size)
    val_meta = index_h5_samples(splits["val"], config.crop_size)
    test_meta = index_h5_samples(splits["test"], config.crop_size)
    meta = {"train": train_meta, "val": val_meta, "test": test_meta}
    save_splits(splits, meta, splits_dir)
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


def make_trial_config(base_config, name, params, log_graph=False):
    return replace(
        base_config,
        **params,
        log_graph=log_graph,
        log_dir=str(RUN_TENSORBOARD_DIR / name),
        best_model_path=str(RUN_MODELS_DIR / f"cnn_{name}_best_model.pt"),
    )


def train_trial(index, name, params, base_config, loaders, input_shape):
    seed_everything(CNN_SPLIT_RANDOM_STATE + index)
    config = make_trial_config(base_config, name, params, log_graph=(index == 1))
    model = ShallowCNN(
        in_channels=input_shape[-1],
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )
    trainer = CNNTrainer(model, config)
    trainer.log_model_graph(input_shape)
    history = trainer.fit(loaders["train"], loaders["val"])
    history.to_csv(RUN_REPORTS_DIR / f"cnn_{name}_history.csv", index=False)

    checkpoint = torch.load(config.best_model_path, map_location=trainer.device)
    return {
        "trial": name,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "best_val_acc": float(checkpoint["val_acc"]),
        "best_model_path": config.best_model_path,
        **params,
    }


def save_prediction_reports(config, params, loaders, meta, input_shape):
    model = ShallowCNN(
        in_channels=input_shape[-1],
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )
    trainer = CNNTrainer(model, config)
    checkpoint = torch.load(config.best_model_path, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from {config.best_model_path}: epoch={checkpoint['epoch']} val_acc={checkpoint['val_acc']:.4f}")

    summaries, by_time = [], []
    writer = trainer.writer
    try:
        for split in ["train", "val", "test"]:
            pred, true = trainer.predict(loaders[split])
            summary, time_rows = summarize_predictions(split, true, pred, meta[split])
            summary.update({"trial": params["trial"], "split_seed": CNN_SPLIT_RANDOM_STATE})
            summaries.append(summary)
            by_time.append(time_rows.assign(trial=params["trial"], split_seed=CNN_SPLIT_RANDOM_STATE))

            predictions = meta[split].copy()
            predictions["y_true"] = true
            predictions["y_pred"] = pred
            predictions["y_true_label"] = predictions["y_true"].map({0: "sound", 1: "damaged"})
            predictions["y_pred_label"] = predictions["y_pred"].map({0: "sound", 1: "damaged"})
            predictions.to_csv(RUN_REPORTS_DIR / f"cnn_{split}_predictions.csv", index=False)

            matrix = confusion_matrix(true, pred, labels=[0, 1])
            pd.DataFrame(
                matrix,
                index=["true_sound", "true_damaged"],
                columns=["pred_sound", "pred_damaged"],
            ).to_csv(RUN_REPORTS_DIR / f"cnn_{split}_confusion_matrix.csv")

            report = classification_report(true, pred, target_names=["sound", "damaged"], zero_division=0)
            report_table = classification_report(true, pred, target_names=["sound", "damaged"], zero_division=0, output_dict=True)
            (RUN_REPORTS_DIR / f"cnn_{split}_classification_report.txt").write_text(report)
            pd.DataFrame(report_table).T.to_csv(RUN_REPORTS_DIR / f"cnn_{split}_classification_report.csv")

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

    pd.DataFrame(summaries).to_csv(RUN_REPORTS_DIR / "cnn_summary.csv", index=False)
    pd.concat(by_time, ignore_index=True).to_csv(RUN_REPORTS_DIR / "cnn_by_timepoint.csv", index=False)


def main():
    seed_everything(CNN_SPLIT_RANDOM_STATE)
    base_config = CNNConfig()
    RUN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    RUN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    loaders, meta, input_shape, splits = build_data(base_config)

    print("Files:", {name: len(paths) for name, paths in splits.items()})
    print("Samples:", {name: len(meta[name]) for name in meta})
    print("Input shape:", input_shape)
    print("Labels:", {name: meta[name]["y"].value_counts().sort_index().to_dict() for name in meta})
    print("Split seed:", CNN_SPLIT_RANDOM_STATE)
    print("Split directory:", RUN_SPLITS_DIR)
    print("Result directory:", RUN_REPORTS_DIR)
    print("Augmentation:", "on" if base_config.augment else "off")

    records = []
    for index, params in enumerate(grid_candidates(), 1):
        name = trial_name(index, params)
        print(f"\nGrid trial {index}: {name} | {params}")
        records.append(train_trial(index, name, params, base_config, loaders, input_shape))

    grid_results = pd.DataFrame(records).sort_values("best_val_acc", ascending=False).reset_index(drop=True)
    grid_results.to_csv(RUN_REPORTS_DIR / "cnn_grid_search_results.csv", index=False)

    best = grid_results.iloc[0].to_dict()
    best_params = {
        "trial": best["trial"],
        "lr": float(best["lr"]),
        "dropout": float(best["dropout"]),
        "weight_decay": float(best["weight_decay"]),
        "conv_channels": tuple(best["conv_channels"]),
    }
    trial_config_params = {k: best_params[k] for k in ["lr", "dropout", "weight_decay", "conv_channels"]}
    best_config = make_trial_config(base_config, best_params["trial"], trial_config_params)
    pd.DataFrame([best]).to_csv(RUN_REPORTS_DIR / "cnn_best_hyperparameters.csv", index=False)
    save_prediction_reports(best_config, best_params, loaders, meta, input_shape)


if __name__ == "__main__":
    main()
