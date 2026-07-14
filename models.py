import torch.nn as nn


class DigitCNNv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2, bias=False), #64, 384
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False), #32, 192
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), #16, 96
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((3, 8))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )

    def forward(self, x):
        out = self.features(x) # B, 64, 3, 8
        out = self.head(out) # B, 50
        return out.reshape((-1, 5, 10)) # B, 5, 10

class DigitCNNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2, bias=False), #64, 384
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False), #32, 192
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), #16, 96
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 5))
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        out = self.features(x) # B, 64, 1, 5
        out = out.squeeze(2) # B, 64, 5
        out = out.transpose(1, 2) # B, 5, 64
        return self.classifier(out) # B, 5, 10

class DigitCNNv2.1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2, bias=False), #64, 384
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False), #32, 192
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), #16, 96
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 5))
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        out = self.features(x) # B, 64, 1, 5
        out = out.squeeze(2) # B, 64, 5
        out = out.transpose(1, 2) # B, 5, 64
        return self.classifier(out) # B, 5, 10
