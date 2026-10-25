"""Central configuration for Project Chimera.

Every tunable value lives here so other modules never hard-code paths,
class names, or hyper-parameters.
"""
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "chimera_cnn.pt"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# -------------------------------------------------------------- classes
# Names must match Quick, Draw! category names exactly (used by the downloader).
CLASS_NAMES = [
    "cat", "dog", "house", "tree", "apple",
    "car", "clock", "star", "fish", "flower",
]
NUM_CLASSES = len(CLASS_NAMES)

# ----------------------------------------------------------- data source
# Quick, Draw! by Google (CC BY 4.0): https://github.com/googlecreativelab/quickdraw-dataset
# "numpy_bitmap" files hold one 28x28 uint8 bitmap per drawing (white ink on black).
QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
MANIFEST_PATH = RAW_DATA_DIR / "manifest.json"

# ---------------------------------------------------------------- image
IMAGE_SIZE = 28          # model input is IMAGE_SIZE x IMAGE_SIZE
IN_CHANNELS = 1          # grayscale
# Normalisation shared by training and inference (pixels first scaled to 0..1).
NORM_MEAN = 0.5
NORM_STD = 0.5

# ------------------------------------------------------- preprocessing
CONTENT_SIZE = 20        # longest side of the drawing inside the 28x28 frame (4 px padding)
INK_THRESHOLD = 25       # pixel value (0-255, after contrast stretch) counted as ink
MIN_INK_PIXELS = 3       # fewer ink pixels than this => image treated as blank
MIN_INK_STRENGTH = 16    # brightest pixel must exceed the background by this much

# ------------------------------------------------------------- training
@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 1e-3
    samples_per_class: int = 5000   # keeps the download/training small
    val_fraction: float = 0.15
    test_fraction: float = 0.15


DEFAULT_TRAIN_CONFIG = TrainConfig()

# ------------------------------------------------------------------ app
TOP_K = 5
