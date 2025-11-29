"""
Descriptor Network Wrappers

Provides CNN-based descriptor extractors with various backbones:
- ResNet (18, 34, 50)
- VGG
- MobileNet
- EfficientNet (if available)

All output L2-normalized descriptor vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


class DescriptorWrapper(nn.Module):
    """
    Generic descriptor wrapper using pretrained CNN backbones.
    
    Args:
        backbone: Backbone architecture ("resnet18", "resnet34", "resnet50", "vgg16", "mobilenet")
        output_dim: Dimension of output descriptor
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone weights (train only head)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        output_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.output_dim = output_dim

        # Get backbone and feature dimension
        self.backbone, feat_dim = self._create_backbone(backbone, pretrained)

        # Descriptor head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, output_dim),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _create_backbone(self, name: str, pretrained: bool):
        """Create backbone network and return it with feature dimension."""

        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            feat_dim = 512

        elif name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            feat_dim = 512

        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            feat_dim = 2048

        elif name == "vgg16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            vgg = models.vgg16(weights=weights)
            backbone = vgg.features
            backbone = nn.Sequential(backbone, nn.AdaptiveAvgPool2d((1, 1)))
            feat_dim = 512

        elif name == "mobilenet":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            mobilenet = models.mobilenet_v2(weights=weights)
            backbone = nn.Sequential(
                mobilenet.features,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            feat_dim = 1280

        else:
            raise ValueError(f"Unknown backbone: {name}")

        return backbone, feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract descriptor from input patches.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            L2-normalized descriptors of shape (B, output_dim)
        """
        feat = self.backbone(x)
        feat = feat.flatten(1)
        desc = self.head(feat)

        return F.normalize(desc, p=2, dim=1)

    def get_embedding_dim(self) -> int:
        """Return the output descriptor dimension."""
        return self.output_dim


class LightweightDescriptor(nn.Module):
    """
    Lightweight CNN descriptor for fast inference.
    
    Simple 5-layer CNN designed for 32x32 patches.
    Much faster than ResNet but less powerful.
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.output_dim = output_dim

        self.features = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 4x4 -> 2x2
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 2x2 -> 1x1
            nn.Conv2d(256, 256, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.flatten(1)
        desc = self.head(feat)
        return F.normalize(desc, p=2, dim=1)

    def get_embedding_dim(self) -> int:
        return self.output_dim


class HardNetDescriptor(nn.Module):
    """
    HardNet-style architecture (Mishchuk et al., 2017).
    
    Designed specifically for local patch descriptors.
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.output_dim = output_dim

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(inplace=True),

            nn.Dropout(0.3),
            nn.Conv2d(128, output_dim, 8, bias=False),
            nn.BatchNorm2d(output_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = self.features(x)
        desc = desc.view(desc.size(0), -1)
        return F.normalize(desc, p=2, dim=1)

    def get_embedding_dim(self) -> int:
        return self.output_dim


def get_descriptor_model(
    name: str = "resnet50",
    output_dim: int = 512,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function for descriptor models.
    
    Args:
        name: Model name ("resnet18", "resnet34", "resnet50", "vgg16",
              "mobilenet", "lightweight", "hardnet")
        output_dim: Output descriptor dimension
        pretrained: Use pretrained weights (for backbone models)
        
    Returns:
        Descriptor model
    """
    if name in ["resnet18", "resnet34", "resnet50", "vgg16", "mobilenet"]:
        return DescriptorWrapper(
            backbone=name,
            output_dim=output_dim,
            pretrained=pretrained,
            **kwargs,
        )
    elif name == "lightweight":
        return LightweightDescriptor(output_dim=output_dim)
    elif name == "hardnet":
        return HardNetDescriptor(output_dim=output_dim)
    else:
        raise ValueError(f"Unknown model: {name}")
