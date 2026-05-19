import torch
import torch.nn as nn
import torch.nn.functional as F


# Unified symbol classes: 10 digits + 7 operators/symbols + 7 lowercase letters = 24 classes
SYMBOL_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',   # 0-9
    '+', '-', '*', '/', '(', ')', '=',                  # 10-16
    'a', 'b', 'c', 'w', 'x', 'y', 'z',                  # 17-23
]
NUM_CLASSES = len(SYMBOL_CLASSES)

# Reverse lookup: character -> index
SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOL_CLASSES)}


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    """Residual block with SE attention."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + x)


class SymbolCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 28x28 -> 14x14
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResBlock(64),

            # Stage 2: 14x14 -> 7x7
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResBlock(128),

            # Stage 3: 7x7 -> 3x3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResBlock(256),

            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
