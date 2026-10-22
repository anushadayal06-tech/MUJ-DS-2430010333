"""model.py - ResNet18 / EfficientNet-B0 (CNNs) and ViT-B/16 (transformer), all via torchvision.
Every model takes RAW [0,1] images and normalizes internally."""
import torch
import torch.nn as nn
from torchvision import models

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ARCHS = ["resnet18", "efficientnet_b0", "vit_b_16"]
DEFAULT_LR = {"resnet18": 1e-4, "efficientnet_b0": 1e-4, "vit_b_16": 2e-5}


class Net(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def get_model(arch="resnet18", pretrained=True):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif arch == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
    else:
        raise ValueError(f"arch must be one of {ARCHS}")
    return Net(m)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
