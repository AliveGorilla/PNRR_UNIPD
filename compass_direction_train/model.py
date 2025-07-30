import torch
import torch.nn as nn
from torchvision import models

class ResNet18Keypoints(nn.Module):
    """
    ResNet18-based CNN model for keypoint regression.

    The output is a vector of 4 normalized coordinates:
        [center_x, center_y, north_x, north_y]
    All coordinates are normalized to [0, 1] relative to image width/height.
    """
    def __init__(self):
        super().__init__()
        # Load a ResNet18 backbone pretrained on ImageNet
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        n_feat = self.backbone.fc.in_features
        # Replace the fully connected layer to output 4 regression values
        self.backbone.fc = nn.Linear(n_feat, 4)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, 4) with normalized coordinates
        """
        return self.backbone(x)