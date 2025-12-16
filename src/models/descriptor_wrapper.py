
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class DescriptorModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        descriptor_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.descriptor_dim = descriptor_dim
        
        if backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.features = nn.Sequential(*list(base.children())[:-1])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.descriptor_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, descriptor_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        descriptors = self.descriptor_head(features)
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return descriptors

def get_descriptor_model(
    name: str = "resnet50",
    descriptor_dim: int = 512,
    pretrained: bool = True,
) -> DescriptorModel:

    return DescriptorModel(
        backbone=name,
        descriptor_dim=descriptor_dim,
        pretrained=pretrained,
    )
