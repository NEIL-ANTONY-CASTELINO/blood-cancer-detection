import argparse

import torch
from PIL import Image

from config import CHECKPOINT_PATH
from dataset import get_transforms
from model import build_model

_model = None
_class_names = None
_device = None


def load_model(checkpoint_path=CHECKPOINT_PATH):
    global _model, _class_names, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=_device)
    _class_names = ckpt["class_names"]
    _model = build_model(num_classes=len(_class_names))
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.to(_device).eval()
    return _model, _class_names


def predict_image(image: Image.Image, checkpoint_path=CHECKPOINT_PATH):
    """Returns (predicted_class: str, probabilities: dict[str, float])."""
    global _model, _class_names
    if _model is None:
        load_model(checkpoint_path)

    transform = get_transforms(train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred_idx = int(torch.tensor(probs).argmax())
    predicted_class = _class_names[pred_idx]
    prob_dict = {cls: p for cls, p in zip(_class_names, probs)}
    return predicted_class, prob_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    args = parser.parse_args()

    image = Image.open(args.image_path)
    predicted_class, probs = predict_image(image, checkpoint_path=args.checkpoint)

    print(f"Prediction: {predicted_class}")
    for cls, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {p:.4f}")


if __name__ == "__main__":
    main()
