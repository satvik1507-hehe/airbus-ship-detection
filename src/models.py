"""The two convolutional networks of the report (Section 4.2).

SimCNN : 4 x (3x3 conv + ReLU + 2x2 max-pool) -> FC(128) -> FC(1), trained from scratch.
         256x256x3 -> 128x128x16 -> 64x64x32 -> 32x32x64 -> 16x16x128 -> 32768 -> 128 -> 1
         4,292,001 trainable parameters (same number as the report).

TLHead : classifier placed on top of a FROZEN ImageNet DenseNet-169.
         1664 pooled features -> BatchNorm -> Dropout -> FC(128) -> FC(1)
         216,577 trainable parameters (same number as the report).

Both output one logit; sigmoid(logit) = P(image contains a ship).
"""
import torch
import torch.nn as nn
from torchvision.models import DenseNet169_Weights, densenet169


class SimCNN(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [3, 16, 32, 64, 128]
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv2d(cin, cout, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):                 # x: [B,3,256,256] in [0,1]
        return self.classifier(self.features(x)).squeeze(1)


class TLHead(nn.Module):
    def __init__(self, in_features=1664):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, feats):             # feats: [B,1664] pooled DenseNet features
        return self.net(feats).squeeze(1)


class DenseNetBackbone(nn.Module):
    """Frozen DenseNet-169 feature extractor followed by global max pooling."""
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        self.features = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1).features
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))
        self.eval()

    @torch.no_grad()
    def forward(self, x):                 # x: [B,3,256,256] RGB in [0,1]
        f = torch.relu(self.features((x - self.mean) / self.std))   # [B,1664,8,8]
        return torch.amax(f, dim=(2, 3))                            # [B,1664]


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("SimCNN trainable parameters:", f"{count_trainable(SimCNN()):,}")
    print("TLHead trainable parameters:", f"{count_trainable(TLHead()):,}")
