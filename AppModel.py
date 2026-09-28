"""
AppModel.py

Single source of truth for the AlzhiNet architecture. train.py and
gradcam_cli.py both import AlzhiNet from here — previously the model class
was defined separately in CNN.py, AppModel.py, and referenced (but not
actually importable) from gradcamCLI.py, which meant the three could drift
out of sync silently. Now there's exactly one definition.

Includes BatchNorm after each conv layer (the original AppModel.py did not
have this) — speeds convergence and improves training stability for a
model trained from scratch rather than fine-tuned from a pretrained backbone.
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
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
