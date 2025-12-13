import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self,):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        logits = self.layers(observation)
        return logits


class ValueFunction(nn.Module):
    def __init__(self,):
        super().__init__()
        