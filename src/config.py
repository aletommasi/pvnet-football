from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Labels
    K_FUTURE_EVENTS: int = 10

    # Split
    TRAIN_FRAC: float = 0.70
    VAL_FRAC: float = 0.15
    TEST_FRAC: float = 0.15
    RANDOM_SEED: int = 42

    # Model
    HIDDEN_DIM: int = 128
    DROPOUT: float = 0.15

    # Training
    BATCH_SIZE: int = 4096
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 12
    EARLY_STOPPING_PATIENCE: int = 3

    # Drive paths (Colab)
    PROJECT_ROOT: Path = Path("/content/drive/MyDrive/pvnet-football")
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
