import torch.nn.functional as F
from collections import deque
import torch.nn as nn
import random
import torch


class ReplayBuffer:
    def __init__(self, size=100):
        self.size = size
        self.buffer = deque()

    def insert(self, transition):
        if len(self.buffer) == self.size:
            self.buffer.popleft()
        self.buffer.append(transition)

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        batch_tensor = torch.tensor(batch)
        return batch_tensor


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.head = nn.Linear(512 * 3 * 3 + 10, 1)

    def forward(self, inputs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = self.layers(inputs)
        x = torch.concat([x, actions], dim=1) # concatenate sideways
        out = self.head(x)
        return out


device = "cuda"
policy = Policy().to(device)
image = torch.randn((1, 3, 255, 255), device=device)
action = F.one_hot(torch.tensor([1], dtype=torch.long, device=device), num_classes=10)
out = policy(image, action)
