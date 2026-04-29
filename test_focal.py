import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

inputs = torch.randn(10) * 10
targets = torch.randint(0, 2, (10,)).float()
criterion = FocalLoss()
print("Valid targets loss:", criterion(inputs, targets).item())

targets_neg = torch.ones(10) * -1
print("Negative targets loss:", criterion(inputs, targets_neg).item())

targets_two = torch.ones(10) * 2
print("Large targets loss:", criterion(inputs, targets_two).item())

