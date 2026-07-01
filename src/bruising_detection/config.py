import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("BRUISING_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"
RANDOM_STATE = 40
TEST_SIZE = 0.10
VAL_SIZE = 0.10
SPATIAL_CROP_SIZE = 0  # 0 or None means no cropping.


@dataclass
class CNNConfig:
    epochs: int = 60
    batch_size: int = 32
    log_dir: str = str(PROJECT_ROOT / "reports" / "tensorboard")
    log_graph: bool = True
    best_model_path: str = str(PROJECT_ROOT / "models" / "best_model.pt")
    early_stop_patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    conv_channels: tuple = (16, 32)
    kernel_size: int = 3
    dropout: float = 0.20
    crop_size: int | tuple | None = SPATIAL_CROP_SIZE
    use_class_weights: bool = True
    num_workers: int = 0
    augment: bool = True
    aug_p: float = 0.50
    noise_std: float = 0.03
    intensity_scale: float = 0.10
    intensity_shift: float = 0.05
    band_dropout_p: float = 0.05
    shift_p: float = 0.75
    max_shift: int = 20
    erase_p: float = 0.25
    erase_size: int = 16
