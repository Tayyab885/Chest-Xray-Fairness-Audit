import torch.nn as nn
from torchvision import models

def build_model(num_labels: int = 14, pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.densenet121(weights=weights)
    in_features = net.classifier.in_features
    net.classifier = nn.Linear(in_features, num_labels)
    return net
