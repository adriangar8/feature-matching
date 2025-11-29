import torch
import torch.nn as nn
import torchvision.models as models

class DescriptorWrapper(nn.Module):
    
    def __init__(self, pretrained=True, output_dim=512):
        
        super().__init__()

        # -- load pretrained ResNet50 --
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # -- remove classification head --
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: [B, 2048, 1, 1]

        # -- map to descriptor vector--
        
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        
        feat = self.backbone(x)      # [B,2048,1,1]
        feat = feat.flatten(1)       # [B,2048]
        desc = self.fc(feat)         # [B,output_dim]
        
        return nn.functional.normalize(desc, dim=1)