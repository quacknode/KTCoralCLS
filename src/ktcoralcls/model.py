from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torchvision import models
EXPERIMENTS: dict[str, str] = {'KTCoralsCLS-ResNet50': 'resnet50', 'KTCoralsCLS-EfficientNet': 'efficientnet_b0', 'KTCoralsCLS-ConvNeXt': 'convnext_tiny', 'KTCoralsCLS-ViT': 'vit_b_16'}
BACKBONES = sorted(set(EXPERIMENTS.values()) | {'resnet18', 'efficientnet_b0'})

def experiment_name(backbone: str) -> str:
    for name, bb in EXPERIMENTS.items():
        if bb == backbone:
            return name
    return f'KTCoralsCLS-{backbone}'

def create_model(num_classes: int, backbone: str='efficientnet_b0', pretrained: bool=True) -> nn.Module:
    weights = 'DEFAULT' if pretrained else None
    if backbone == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == 'convnext_tiny':
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif backbone == 'vit_b_16':
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f'Unsupported backbone: {backbone}. Known: {sorted(EXPERIMENTS.values())}')
    return model

def _backbone_and_dim(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
    weights = 'DEFAULT' if pretrained else None
    if backbone == 'convnext_tiny':
        model = models.convnext_tiny(weights=weights)
        dim = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        return (model, dim)
    if backbone == 'resnet50':
        model = models.resnet50(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return (model, dim)
    if backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        dim = model.classifier[1].in_features
        model.classifier[1] = nn.Identity()
        return (model, dim)
    if backbone == 'vit_b_16':
        model = models.vit_b_16(weights=weights)
        dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
        return (model, dim)
    raise ValueError(f'Unsupported backbone for multi-task: {backbone}')

class MultiTaskCoralClassifier(nn.Module):

    def __init__(self, num_genus: int, num_growth: int, backbone: str='convnext_tiny', pretrained: bool=True) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.encoder, dim = _backbone_and_dim(backbone, pretrained)
        self.genus_head = nn.Linear(dim, num_genus)
        self.growth_head = nn.Linear(dim, num_growth)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        if feats.ndim > 2:
            feats = feats.flatten(1)
        return (self.genus_head(feats), self.growth_head(feats))

def create_multitask_model(num_genus: int, num_growth: int, backbone: str='convnext_tiny', pretrained: bool=True) -> MultiTaskCoralClassifier:
    return MultiTaskCoralClassifier(num_genus, num_growth, backbone=backbone, pretrained=pretrained)

def load_checkpoint(path: str | Path, device: torch.device, num_classes: int, backbone: str | None=None) -> tuple[nn.Module, dict]:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    bb = backbone or ckpt.get('backbone')
    if bb is None:
        raise ValueError(f'Checkpoint {path} has no backbone; pass backbone= explicitly')
    model = create_model(num_classes, backbone=bb, pretrained=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return (model, ckpt)
