import torch.nn as nn
import torch

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
            nn.Linear(64 * 3 * 8, 64),
            nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(64, 50)
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

class DigitCNNv2p1(nn.Module):
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

class DigitCNNCompact(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((2, 10)),
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(32 * 2 * 2, 10)

    def forward(self, x):
        x = self.features(x)       # (B, 64, 2, 10)
        x = self.head(x)           # (B, 32, 2, 10)

        x = x.reshape(
            x.size(0), 32, 2, 5, 2
        )                          # (B, 32, 2, 5, 2)

        x = x.permute(0, 3, 1, 2, 4)
        x = x.flatten(2)           # (B, 5, 128)

        return self.classifier(x)  # (B, 5, 10)

class PositionHeadsCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 10)),
            nn.Flatten(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(64 * 4 * 10, 10)
            for _ in range(5)
        ])

    def forward(self, x):
        features = self.features(x)

        return torch.stack(
            [head(features) for head in self.heads],
            dim=1,
        )

class SequenceCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 24)),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.pool = nn.AdaptiveAvgPool1d(5)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.cnn(x)          # (B, 128, 1, 24)
        x = x.squeeze(2)         # (B, 128, 24)
        x = x.transpose(1, 2)    # (B, 24, 128)

        x = self.encoder(x)      # (B, 24, 128)

        x = x.transpose(1, 2)    # (B, 128, 24)
        x = self.pool(x)         # (B, 128, 5)
        x = x.transpose(1, 2)    # (B, 5, 128)

        return self.classifier(x)  # (B, 5, 10)

class LevelConditionedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((2, 10)),
            nn.Flatten(),
        )

        self.level_embedding = nn.Embedding(5, 16)

        self.head = nn.Sequential(
            nn.Linear(128 * 2 * 10 + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 5 * 10),
            nn.Unflatten(1, (5, 10)),
        )

    def forward(self, x, level):
        image_features = self.features(x)
        level_features = self.level_embedding(level)

        features = torch.cat(
            [image_features, level_features],
            dim=1,
        )

        return self.head(features)