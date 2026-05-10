import torch
import torch.nn as nn
from torchvision import models


class DualHeadModel(nn.Module):
    """
    Transfer learning backbone with two output heads:
      - fc         → main task: 120-breed fine-grained classification
      - group_head → auxiliary task: 16-group coarse classification

    Training with hierarchical supervision (breed loss + weighted group loss) forces
    the backbone to first learn visually coherent group-level features (e.g. the long
    snout and floppy ears common to all scent hounds) before specialising in the subtle
    inter-breed differences. This is the core idea behind hierarchical fine-grained
    recognition and consistently improves convergence on small datasets.

    Both heads share the same feature extractor: only one forward pass is needed.
    """

    def __init__(self, architecture: str, num_classes: int, num_groups: int):
        super().__init__()
        self.arch_name = architecture   # kept as plain str, not an nn.Module

        if architecture == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            _freeze_backbone(base)
            in_features = base.fc.in_features
            # Copy sub-modules as named attributes so unfreeze_top_layers works by name
            self.conv1   = base.conv1
            self.bn1     = base.bn1
            self.relu    = base.relu
            self.maxpool = base.maxpool
            self.layer1  = base.layer1
            self.layer2  = base.layer2
            self.layer3  = base.layer3
            self.layer4  = base.layer4  # unfrozen in phase 2
            self.avgpool = base.avgpool

        elif architecture == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            _freeze_backbone(base)
            in_features = base.fc.in_features
            self.conv1   = base.conv1
            self.bn1     = base.bn1
            self.relu    = base.relu
            self.maxpool = base.maxpool
            self.layer1  = base.layer1
            self.layer2  = base.layer2
            self.layer3  = base.layer3
            self.layer4  = base.layer4
            self.avgpool = base.avgpool

        elif architecture == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            _freeze_backbone(base)
            in_features   = base.classifier[1].in_features
            self.features = base.features
            self.avgpool  = base.avgpool

        elif architecture == "efficientnet_v2_s":
            # EfficientNetV2-S: 84.2% ImageNet top-1 vs ResNet50's 80.9%.
            # Trained with progressive learning on larger images; pretrained features are
            # significantly richer than ResNet50, which should help fine-grained discrimination.
            base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            _freeze_backbone(base)
            in_features   = base.classifier[1].in_features   # 1280
            self.features = base.features   # unfrozen [-3:] in phase 2
            self.avgpool  = base.avgpool

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Main classification head: 120 dog breeds
        self.fc         = nn.Linear(in_features, num_classes)
        # Auxiliary head: 16 visual groups (terrier, hound, retriever, …)
        # Shares the feature vector with fc → zero extra compute
        self.group_head = nn.Linear(in_features, num_groups)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the frozen backbone to produce a flat feature vector."""
        if self.arch_name in ("resnet50", "resnet18"):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
        else:  # efficientnet_b0
            x = self.features(x)
            x = self.avgpool(x)
        return x.flatten(1)   # (B, in_features)

    def forward(self, x: torch.Tensor):
        feat = self._extract_features(x)
        # Return both heads; the caller combines their losses during training
        return self.fc(feat), self.group_head(feat)


def get_model(architecture: str, num_classes: int, num_groups: int) -> DualHeadModel:
    return DualHeadModel(architecture, num_classes, num_groups)


def unfreeze_top_layers(model: DualHeadModel, architecture: str) -> None:
    """Unfreeze the last convolutional block for discriminative fine-tuning (phase 2)."""
    if architecture in ("resnet50", "resnet18"):
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif architecture in ("efficientnet_b0", "efficientnet_v2_s"):
        for param in model.features[-3:].parameters():
            param.requires_grad = True


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter so only the newly added heads train in phase 1."""
    for param in model.parameters():
        param.requires_grad = False
