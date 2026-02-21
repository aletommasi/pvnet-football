from __future__ import annotations
import torch.nn as nn

class PVNet(nn.Module):
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)
