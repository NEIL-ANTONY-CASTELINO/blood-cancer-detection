"""
Downloads a couple of parquet shards (not the whole 6.5GB dataset) from the
C-NMC 2019 leukemia dataset on Hugging Face and materializes a balanced,
resized image split on disk:

    data/processed/{train,val,test}/{cancerous,non_cancerous}/*.jpg

The dataset's shards are grouped by class: shard 0 is all "cancer" and
shard 12 (the last of 13) is all "healthy" — so downloading just those two
(~500MB total) is enough to build a balanced dataset without pulling
everything.

Run once before training. Use --per-class to cap how many images per class
are materialized.
"""
import argparse
import io
import json
import random
from collections import defaultdict

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from config import (
    DATA_CACHE_DIR,
    HF_DATASET_NAME,
    IMAGE_SIZE,
    LABELS_PATH,
    ROOT,
    SEED,
    TEST_FRAC,
    TRAIN_FRAC,
)

PROCESSED_DIR = ROOT / "data" / "processed"

# raw HF `label` column -> our class name, and which shard holds each class
LABEL_MAP = {"cancer": "cancerous", "healthy": "non_cancerous"}
SHARDS_BY_LABEL = {"cancer": 0, "healthy": 12}
NUM_SHARDS = 13
CLASS_NAMES = sorted(set(LABEL_MAP.values()))  # alphabetical, matches ImageFolder


def decode_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes))


def load_shard(shard_index: int) -> pd.DataFrame:
    path = hf_hub_download(
        HF_DATASET_NAME,
        f"data/train-{shard_index:05d}-of-{NUM_SHARDS:05d}.parquet",
        repo_type="dataset",
    )
    return pd.read_parquet(path, columns=["image", "label", "original_image_name"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-class",
        type=int,
        default=400,
        help="Max images per class to materialize (0 = use everything in the shard).",
    )
    args = parser.parse_args()

    random.seed(SEED)
    cap = args.per_class if args.per_class > 0 else None

    buckets = {}
    for raw_label, shard_index in SHARDS_BY_LABEL.items():
        cls = LABEL_MAP[raw_label]
        print(f"Fetching shard {shard_index} for class '{cls}' ({raw_label})...")
        df = load_shard(shard_index)
        df = df[df["label"] == raw_label]
        if cap is not None and len(df) > cap:
            df = df.sample(n=cap, random_state=SEED)
        buckets[cls] = df.to_dict("records")
        print(f"  using {len(buckets[cls])} '{cls}' images")

    manifest = []
    for cls, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * (1 - TRAIN_FRAC - TEST_FRAC))
        splits = (
            ["train"] * n_train
            + ["val"] * n_val
            + ["test"] * (n - n_train - n_val)
        )
        for row, split in zip(items, splits):
            img = decode_image(row["image"]["bytes"]).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

            out_dir = PROCESSED_DIR / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = str(row.get("original_image_name") or f"{cls}_{len(manifest)}").replace("/", "_")
            if not fname.lower().endswith((".jpg", ".jpeg")):
                fname += ".jpg"
            out_path = out_dir / fname
            img.save(out_path, "JPEG", quality=90)

            manifest.append({"path": str(out_path.relative_to(ROOT)), "class": cls, "split": split})

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_CACHE_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(LABELS_PATH, "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    print(f"\nDone. {len(manifest)} images written under {PROCESSED_DIR}")
    print(f"Manifest: {DATA_CACHE_DIR / 'manifest.json'}")
    print(f"Labels:   {LABELS_PATH}")


if __name__ == "__main__":
    main()
