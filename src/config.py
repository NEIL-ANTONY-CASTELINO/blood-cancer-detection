from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = ROOT / "data" / "cache"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"
LABELS_PATH = DATA_CACHE_DIR / "labels.json"

HF_DATASET_NAME = "dwb2023/cnmc-leukemia-2019"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 42

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
