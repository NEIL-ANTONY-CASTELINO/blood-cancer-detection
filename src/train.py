import argparse
import json
import time

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import CHECKPOINT_DIR, CHECKPOINT_PATH, SEED
from dataset import get_dataloader, load_class_names
from model import build_model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = load_class_names()
    train_loader = get_dataloader("train", batch_size=args.batch_size)
    val_loader = get_dataloader("val", batch_size=args.batch_size)

    model = build_model(num_classes=len(class_names), freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                CHECKPOINT_PATH,
            )
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f}) to {CHECKPOINT_PATH}")

    print("\nBest validation accuracy:", best_val_acc)
    print("\nFinal-epoch validation report:")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(val_labels, val_preds))

    with open(CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
