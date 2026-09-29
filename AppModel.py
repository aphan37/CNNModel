"""
AppModel.py

Single source of truth for the AlzhiNet architecture. train.py and
gradcam_cli.py both import AlzhiNet from here — previously the model class
was defined separately in CNN.py, AppModel.py, and referenced (but not
actually importable) from gradcamCLI.py, which meant the three could drift
out of sync silently. Now there's exactly one definition.

Includes two changes vs. the original AppModel.py:
  1. BatchNorm after each conv layer — speeds convergence and improves
     training stability for a model trained from scratch.
  2. Global average pooling before the FC head, instead of a flatten.
     The original flattened the full 256x28x28 feature map (200,704
     values) straight into a 1024-unit FC layer -- a ~205 MILLION
     parameter layer, by far the majority of the entire model's weights.
     That's expensive to train (it OOM'd outright in a 4GB-RAM sandbox
     while smoke-testing this), slow on a modest laptop GPU, and adds a
     lot of overfitting risk for a dataset with limited patients per
     class. Global average pooling collapses each of the 256 channels to
     a single value (a standard technique used in ResNet, DenseNet, and
     most modern CNNs for exactly this reason), so the FC layer becomes
     256 -> 1024 instead of 200,704 -> 1024: roughly 800x fewer
     parameters in that layer, with no loss of the spatial feature
     extraction done by the conv blocks before it.
"""

import torch.nn as nn
import torch.nn.functional as F


class AlzhiNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (B, 256, H, W) -> (B, 256, 1, 1), any input size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
