from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        for param in backbone.parameters():
            param.requires_grad_(False)

        modules = list(backbone.children())[:-1]  # output layer 제거
        self.backbone = nn.Sequential(*modules)
        self.embed = nn.Linear(backbone.fc.in_features, embed_dim)  # embedding dim으로 변환하는 Output layer 추가.

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embed(x)

        return x