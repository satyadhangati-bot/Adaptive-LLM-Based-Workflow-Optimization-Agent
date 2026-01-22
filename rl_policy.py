import torch
from torch import nn, optim

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)

class RLPolicy:
    def __init__(self):
        self.model = PolicyNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def update(self, state, action, reward):
        logits = self.model(state)
        loss = -logits[0, action] * reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
