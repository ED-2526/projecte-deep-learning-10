import torch
import torch.nn as nn
from torchvision import models


class DualHeadModel(nn.Module):
    def __init__(self, num_classes: int, num_groups: int, dropout: float = 0.0, pretrained: bool = True):
        super().__init__()
        base = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        )
        for param in base.parameters():
            param.requires_grad = False

        in_features   = base.classifier[1].in_features  # 1280
        self.features = base.features
        self.avgpool  = base.avgpool
        self.dropout  = nn.Dropout(p=dropout)
        self.fc         = nn.Linear(in_features, num_classes)
        self.group_head = nn.Linear(in_features, num_groups)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x), self.group_head(x)


def get_model(num_classes: int, num_groups: int, dropout: float = 0.0, pretrained: bool = True) -> DualHeadModel:
    return DualHeadModel(num_classes, num_groups, dropout=dropout, pretrained=pretrained)


def unfreeze_top_layers(model: DualHeadModel, num_blocks: int = 3) -> None:
    for param in model.features[-num_blocks:].parameters():
        param.requires_grad = True
