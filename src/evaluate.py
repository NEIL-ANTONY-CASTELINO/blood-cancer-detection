import argparse

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from config import CHECKPOINT_DIR, CHECKPOINT_PATH
from dataset import get_dataloader
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    loader = get_dataloader(args.split, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    print(f"Checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})\n")
    print(f"{args.split} classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    out_path = CHECKPOINT_DIR / f"confusion_matrix_{args.split}.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved confusion matrix plot to {out_path}")


if __name__ == "__main__":
    main()
