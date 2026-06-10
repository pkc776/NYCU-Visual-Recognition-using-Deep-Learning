import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(ModifiedResNet50, self).__init__()

        # Load pretrained ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Modify the fully connected layer
        # The original fc is: Linear(in_features=2048,
        #                            out_features=1000, bias=True)
        in_features = self.backbone.fc.in_features

        # We replace it with a Sequential block containing Dropout and a
        # new Linear layer. This modification satisfies the homework
        # constraints to adjust the backbone
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
