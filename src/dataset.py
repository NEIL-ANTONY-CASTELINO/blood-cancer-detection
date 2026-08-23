import json

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABELS_PATH,
    NUM_WORKERS,
    ROOT,
)

PROCESSED_DIR = ROOT / "data" / "processed"


def get_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_class_names():
    with open(LABELS_PATH) as f:
        return json.load(f)


def get_dataloader(split: str, batch_size: int = BATCH_SIZE, shuffle: bool = None):
    assert split in ("train", "val", "test")
    split_dir = PROCESSED_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"{split_dir} not found. Run `python src/prepare_data.py` first."
        )

    dataset = ImageFolder(str(split_dir), transform=get_transforms(train=(split == "train")))

    class_names = load_class_names()
    if dataset.classes != class_names:
        raise ValueError(
            f"ImageFolder class order {dataset.classes} does not match "
            f"expected {class_names} (from {LABELS_PATH})"
        )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
