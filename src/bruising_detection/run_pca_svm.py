from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.patches import FancyBboxPatch

from .config import REPORTS_DIR, SPLITS_DIR, PROJECT_ROOT

RANDOM_STATE = 42
N_COMPONENTS = 5
SPATIAL_CROP_SIZE = 10
COMPONENT_GRID = [3, 6, 8, 10]
KERNEL_GRID = ["linear", "rbf", "poly", "sigmoid"]
LABELS = [0, 1]
LABEL_NAMES = ["sound", "damaged"]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def split_paths(split):
    return sorted((SPLITS_DIR / split).glob("MICROTEC_*_processed_boxes.h5"))


def crop_cube(cube, crop_size=SPATIAL_CROP_SIZE):
    if crop_size is None:
        return cube
    h, w = cube.shape[:2]
    ch, cw = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    return cube[r0 : r0 + ch, c0 : c0 + cw, :]


def crop_size_label(crop_size=SPATIAL_CROP_SIZE):
    if crop_size is None:
        return "full box"
    ch, cw = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    return f"center {ch}x{cw}"


def draw_model_pipelines(path=None, n_components=N_COMPONENTS):
    pipelines = {
        "Mean-spectrum model": [crop_size_label(), "average rows/cols", f"PCA: {n_components}", "SVM"],
        "Flattened crop model": [crop_size_label(), f"pixel PCA: {n_components}", f"flatten: {SPATIAL_CROP_SIZE}x{SPATIAL_CROP_SIZE}x{n_components}", "SVM"],
    }
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.8))
    for ax, (title, steps) in zip(axes, pipelines.items()):
        ax.set_xlim(0, len(steps))
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, loc="left")
        for i, step in enumerate(steps):
            box = FancyBboxPatch((i + 0.05, 0.32), 0.8, 0.34, boxstyle="round,pad=0.03", facecolor="white", edgecolor="black")
            ax.add_patch(box)
            ax.text(i + 0.45, 0.49, step, ha="center", va="center")
            if i < len(steps) - 1:
                ax.annotate("", xy=(i + 1.02, 0.49), xytext=(i + 0.88, 0.49), arrowprops={"arrowstyle": "->"})
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


def label_for(timepoint, ds_name):
    name = ds_name.lower()
    if timepoint == "t0" or name.startswith("sound"):
        return 0
    if name.startswith("bruised"):
        return 1
    return None


def load_split(paths):
    crops, y, rows = [], [], []
    for path in paths:
        with h5py.File(path, "r") as f:
            for group_name, group in f.items():
                timepoint = group_name.split("_")[-1]
                for ds_name, ds in group.items():
                    if not isinstance(ds, h5py.Dataset):
                        continue
                    label = label_for(timepoint, ds_name)
                    if label is None:
                        continue
                    cube = ds[()]
                    crop = crop_cube(cube).astype(np.float32, copy=False)
                    crops.append(crop)
                    y.append(label)
                    rows.append(
                        {
                            "path": str(path),
                            "file": path.name,
                            "group": group_name,
                            "timepoint": timepoint,
                            "dataset": ds_name,
                            "y": label,
                            "label": LABEL_NAMES[label],
                            "box_shape": tuple(cube.shape[:2]),
                            "crop_shape": tuple(crop.shape[:2]),
                            "bands": int(crop.shape[-1]),
                        }
                    )
    return np.stack(crops), np.asarray(y), pd.DataFrame(rows)


def mean_features(crops):
    return crops.mean(axis=(1, 2), dtype=np.float32)


def flat_pca_features(train, val, test, n_components=N_COMPONENTS):
    n = min(n_components, train.shape[-1])
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    pca.fit(train.reshape(-1, train.shape[-1]))

    def transform(crops):
        scores = pca.transform(crops.reshape(-1, crops.shape[-1]))
        return scores.reshape(crops.shape[0], -1).astype(np.float32, copy=False)

    return transform(train), transform(val), transform(test), pca


def make_svm(with_pca=False, n_components=N_COMPONENTS, kernel="linear"):
    steps = [("scale", StandardScaler())]
    if with_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)))
    steps.append(("svm", SVC(kernel=kernel, class_weight="balanced", random_state=RANDOM_STATE)))
    return Pipeline(steps)


def summarize(split, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=LABELS).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "split": split,
        "n": int(len(y_true)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "n_true_positive_samples": int((y_true == 1).sum()),
        "n_predicted_positive_samples": int((y_pred == 1).sum()),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }


def by_timepoint(split, y_true, y_pred, meta):
    scored = meta.copy()
    scored["y_true"] = y_true
    scored["y_pred"] = y_pred
    rows = []
    for tp, group in scored.groupby("timepoint", sort=False):
        rows.append({"split": split, "timepoint": tp, **summarize(split, group["y_true"].to_numpy(), group["y_pred"].to_numpy())})
    return pd.DataFrame(rows)


def save_split_reports(split, y_true, y_pred, meta):
    pred = meta.copy()
    pred["y_true"] = y_true
    pred["y_pred"] = y_pred
    pred["y_true_label"] = pred["y_true"].map({0: "sound", 1: "damaged"})
    pred["y_pred_label"] = pred["y_pred"].map({0: "sound", 1: "damaged"})
    pred.to_csv(REPORTS_DIR / f"pca_svm_{split}_predictions.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(matrix, index=["true_sound", "true_damaged"], columns=["pred_sound", "pred_damaged"]).to_csv(
        REPORTS_DIR / f"pca_svm_{split}_confusion_matrix.csv"
    )

    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0)
    report_table = classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0, output_dict=True)
    (REPORTS_DIR / f"pca_svm_{split}_classification_report.txt").write_text(report)
    pd.DataFrame(report_table).T.to_csv(REPORTS_DIR / f"pca_svm_{split}_classification_report.csv")


def grid_search_pca_svm(train_crops, val_crops, test_crops, y_train, y_val, y_test, X_train_mean, X_val_mean, X_test_mean):
    rows = []
    for n_comp in COMPONENT_GRID:
        n_mean = min(n_comp, X_train_mean.shape[0], X_train_mean.shape[1])
        Xtr_f, Xva_f, Xte_f, _ = flat_pca_features(train_crops, val_crops, test_crops, n_comp)
        for kernel in KERNEL_GRID:
            candidates = [
                ("mean spectrum PCA+SVM", make_svm(True, n_mean, kernel), X_train_mean, X_val_mean, X_test_mean, n_mean),
                ("flattened crop PCA-pixels+SVM", Pipeline([("scale", StandardScaler()), ("svm", SVC(kernel=kernel, class_weight="balanced", random_state=RANDOM_STATE))]), Xtr_f, Xva_f, Xte_f, Xtr_f.shape[1]),
            ]
            for model_name, model, Xtr, Xva, Xte, features in candidates:
                model.fit(Xtr, y_train)
                rows.append({
                    "model": model_name, "kernel": kernel, "components": n_comp,
                    "raw_features": Xtr.shape[1], "svm_features": features,
                    "val_accuracy": round(accuracy_score(y_val, model.predict(Xva)), 4),
                    "test_accuracy": round(accuracy_score(y_test, model.predict(Xte)), 4),
                })
    return pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {split: split_paths(split) for split in ["train", "val", "test"]}
    train_crops, y_train, train_meta = load_split(paths["train"])
    val_crops, y_val, val_meta = load_split(paths["val"])
    test_crops, y_test, test_meta = load_split(paths["test"])

    X_train_mean, X_val_mean, X_test_mean = map(mean_features, [train_crops, val_crops, test_crops])
    draw_model_pipelines(FIGURES_DIR / "pca_svm_model_pipelines.png", N_COMPONENTS)
    X_train_flat, X_val_flat, X_test_flat, pixel_pca = flat_pca_features(train_crops, val_crops, test_crops)

    mean_model = make_svm(with_pca=True, n_components=min(N_COMPONENTS, X_train_mean.shape[0], X_train_mean.shape[1]))
    flat_model = make_svm(with_pca=False)
    mean_model.fit(X_train_mean, y_train)
    flat_model.fit(X_train_flat, y_train)

    base_rows = []
    for name, model, sets in [
        ("mean spectrum PCA+SVM", mean_model, [("val", X_val_mean, y_val), ("test", X_test_mean, y_test)]),
        ("flattened crop PCA-pixels+SVM", flat_model, [("val", X_val_flat, y_val), ("test", X_test_flat, y_test)]),
    ]:
        for split, X, y in sets:
            base_rows.append({"model": name, **summarize(split, y, model.predict(X))})
    pd.DataFrame(base_rows).to_csv(REPORTS_DIR / "pca_svm_base_model_summary.csv", index=False)

    tuning = grid_search_pca_svm(train_crops, val_crops, test_crops, y_train, y_val, y_test, X_train_mean, X_val_mean, X_test_mean)
    tuning.to_csv(REPORTS_DIR / "pca_svm_tuning_results.csv", index=False)

    winner = tuning.iloc[0]
    if winner["model"].startswith("mean"):
        X_sets = {"train": X_train_mean, "val": X_val_mean, "test": X_test_mean}
        model = make_svm(True, min(int(winner["components"]), X_train_mean.shape[0], X_train_mean.shape[1]), winner["kernel"])
    else:
        Xtr, Xva, Xte, pixel_pca = flat_pca_features(train_crops, val_crops, test_crops, int(winner["components"]))
        X_sets = {"train": Xtr, "val": Xva, "test": Xte}
        model = Pipeline([("scale", StandardScaler()), ("svm", SVC(kernel=winner["kernel"], class_weight="balanced", random_state=RANDOM_STATE))])
    model.fit(X_sets["train"], y_train)
    y_sets = {"train": y_train, "val": y_val, "test": y_test}
    meta_sets = {"train": train_meta, "val": val_meta, "test": test_meta}

    summaries, time_tables = [], []
    for split in ["train", "val", "test"]:
        pred = model.predict(X_sets[split])
        summaries.append({"model": winner["model"], "kernel": winner["kernel"], "components": int(winner["components"]), **summarize(split, y_sets[split], pred)})
        time_tables.append(by_timepoint(split, y_sets[split], pred, meta_sets[split]))
        save_split_reports(split, y_sets[split], pred, meta_sets[split])

    pd.DataFrame(summaries).to_csv(REPORTS_DIR / "pca_svm_summary.csv", index=False)
    pd.concat(time_tables, ignore_index=True).to_csv(REPORTS_DIR / "pca_svm_by_timepoint.csv", index=False)
    joblib.dump({"model": model, "winner": winner.to_dict(), "pixel_pca": pixel_pca if not winner["model"].startswith("mean") else None}, PROJECT_ROOT / "models" / "pca_svm_best_model.joblib")
    print(pd.DataFrame(summaries))


if __name__ == "__main__":
    main()
