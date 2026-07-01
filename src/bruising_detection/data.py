import h5py
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import RANDOM_STATE, SPATIAL_CROP_SIZE


def seed_everything(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Exact same split method as pca_svm_processed_box_split.ipynb.
def split_files(paths, test_size=0.2, val_size=0.2, random_state=40):
    train_val, test = train_test_split(sorted(paths), test_size=test_size, random_state=random_state, shuffle=True)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state, shuffle=True)
    return {"train": train, "val": val, "test": test}




def materialize_split_data(splits, out_dir):
    materialized = {}
    for split, paths in splits.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        expected_names = {path.name for path in paths}
        for existing in split_dir.glob("*.h5"):
            if existing.name not in expected_names:
                existing.unlink()

        split_paths = []
        for path in paths:
            target = split_dir / path.name
            if not target.exists() or target.stat().st_size != path.stat().st_size:
                shutil.copy2(path, target)
            split_paths.append(target)
        materialized[split] = split_paths
    return materialized


def save_splits(splits, meta=None, out_dir=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, paths in splits.items():
        pd.DataFrame({
            "file": [path.name for path in paths],
            "path": [str(path) for path in paths],
        }).to_csv(out_dir / f"{split}_files.csv", index=False)

    if meta is None:
        return
    for split, frame in meta.items():
        frame.to_csv(out_dir / f"{split}_samples.csv", index=False)

def centered_crop_bounds(spatial_shape, crop_size=SPATIAL_CROP_SIZE):
    height, width = spatial_shape[:2]
    if crop_size is None or crop_size == 0:
        return 0, height, 0, width
    crop_h, crop_w = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    if crop_h <= 0 or crop_w <= 0 or crop_h > height or crop_w > width:
        raise ValueError(f"Bad crop size {crop_size} for spatial shape {spatial_shape}")
    row0 = (height - crop_h) // 2
    col0 = (width - crop_w) // 2
    return row0, row0 + crop_h, col0, col0 + crop_w


def label_for(timepoint, ds_name):
    name = ds_name.lower()
    if timepoint == "t0" or name.startswith("sound"):
        return 0
    if name.startswith("bruised"):
        return 1
    return None


def index_h5_samples(paths, crop_size=SPATIAL_CROP_SIZE):
    rows = []
    for path in tqdm(paths, desc="Indexing h5 files"):
        with h5py.File(path, "r") as f:
            for group_name, group in f.items():
                timepoint = group_name.split("_")[-1]
                for ds_name, ds in group.items():
                    if not isinstance(ds, h5py.Dataset):
                        continue
                    y = label_for(timepoint, ds_name)
                    if y is None:
                        continue
                    row0, row1, col0, col1 = centered_crop_bounds(ds.shape[:2], crop_size)
                    rows.append({
                        "path": str(path),
                        "file": path.name,
                        "group": group_name,
                        "timepoint": timepoint,
                        "dataset": ds_name,
                        "y": y,
                        "label": "damaged" if y else "sound",
                        "box_shape": ds.shape[:2],
                        "crop_shape": (row1 - row0, col1 - col0),
                        "bands": ds.shape[2],
                    })
    return pd.DataFrame(rows)


class LazyH5CropDataset(Dataset):
    def __init__(self, meta, crop_size=SPATIAL_CROP_SIZE, mean=None, std=None, augment=None):
        self.meta = meta.reset_index(drop=True)
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.y = torch.tensor(self.meta["y"].to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        with h5py.File(row.path, "r") as f:
            ds = f[row.group][row.dataset]
            r0, r1, c0, c1 = centered_crop_bounds(ds.shape[:2], self.crop_size)
            cube = ds[r0:r1, c0:c1, :]
        x = torch.from_numpy(np.ascontiguousarray(cube)).permute(2, 0, 1).float()
        if self.mean is not None:
            x = (x - self.mean) / self.std
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y[idx]


def compute_band_stats(dataset, batch_size=8, num_workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    total = sq_total = None
    count = 0
    for x, _ in tqdm(loader, desc="Computing train mean/std"):
        dims = (0, 2, 3)
        total = x.sum(dims) if total is None else total + x.sum(dims)
        sq_total = (x * x).sum(dims) if sq_total is None else sq_total + (x * x).sum(dims)
        count += x.size(0) * x.size(2) * x.size(3)
    mean = (total / count).view(-1, 1, 1)
    std = (sq_total / count - mean.view(-1).pow(2)).clamp_min(1e-6).sqrt().view(-1, 1, 1)
    return mean, std
