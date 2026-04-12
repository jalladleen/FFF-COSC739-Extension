"""Standalone GenerativeCanary module.

Shared between:
  - hpc_ensemble_1000.ipynb
  - train_ensemble_parallel.py
  - evaluate_ensemble.py

Identical to cell-5 of generative_canary.ipynb. Do not drift — if one changes, all must.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerativeCanary(nn.Module):
    """Lightweight generator that produces unique 80x80x3 canary patches.

    Input:  z = [eta || b] in R^(z_dim + 4)
            eta ~ N(0, I)  (random noise, default 128-dim)
            b = (x1, y1, x2, y2)  (normalized bbox coords)
    Output: canary patch of shape (3, 80, 80) with pixels in [0, 1]
    """

    def __init__(self, z_dim=128, bbox_dim=4, canary_size=80):
        super().__init__()
        self.z_dim = z_dim
        self.bbox_dim = bbox_dim
        self.canary_size = canary_size
        input_dim = z_dim + bbox_dim  # 132

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 16, 8, 8)
        x = self.deconv(x)
        x = F.interpolate(x, size=(self.canary_size, self.canary_size),
                          mode='bilinear', align_corners=False)
        return x


if __name__ == '__main__':
    g = GenerativeCanary()
    n = sum(p.numel() for p in g.parameters())
    print(f"GenerativeCanary parameters: {n:,}")
    z = torch.randn(1, 132)
    out = g(z)
    print(f"Output shape: {tuple(out.shape)}")
