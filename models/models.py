import torch.nn as nn
from torchvision import models


def get_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        _freeze_backbone(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        _freeze_backbone(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        _freeze_backbone(model)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


def unfreeze_top_layers(model: nn.Module, architecture: str) -> None:
    """Unfreeze the last convolutional block so it can be fine-tuned."""
    if architecture in ("resnet50", "resnet18"):
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif architecture == "efficientnet_b0":
        for param in model.features[-3:].parameters():
            param.requires_grad = True


def _freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
