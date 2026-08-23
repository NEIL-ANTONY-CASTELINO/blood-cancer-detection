import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """ResNet18 pretrained on ImageNet, fine-tuned for binary classification."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    # the new head is always trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
