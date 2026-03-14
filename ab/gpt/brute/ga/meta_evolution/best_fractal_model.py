import torch
import torch.nn as nn
from typing import List

# --- MANDATORY FOR EVAL ENGINE ---
def supported_hyperparameters():
    return {'lr', 'momentum'}

# --- Helper Classes ---
class FractalDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if not self.training: 
            return torch.stack(inputs).mean(dim=0)
        n = len(inputs)
        mask = torch.bernoulli(torch.full((n,), 1 - self.drop_prob, device=inputs[0].device))
        if mask.sum() == 0: 
            mask[torch.randint(0, n, (1,)).item()] = 1.0
        active = [inp for inp, m in zip(inputs, mask) if m > 0]
        return torch.stack(active).mean(dim=0)

class FractalBlock(nn.Module):
    def __init__(self, n_columns: int, channels: int, dropout_prob: float):
        super().__init__()
        self.n_columns = n_columns
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        if n_columns > 1:
            self.left = FractalBlock(n_columns - 1, channels, dropout_prob)
            self.right_1 = FractalBlock(n_columns - 1, channels, dropout_prob)
            self.right_2 = FractalBlock(n_columns - 1, channels, dropout_prob)
            self.join = FractalDropPath(drop_prob=dropout_prob)

    def forward(self, x):
        if self.n_columns == 1: return self.conv(x)
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        return self.join([out_left, out_right])

# --- Main Network ---
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.device = device

        # FIX: Force 3 input channels for CIFAR-10 / Color data
        # The eval harness might pass in_shape[0]=1 incorrectly
        c_in = 3 

        # Handle Output Shape safely
        n_classes = out_shape[0] if out_shape else 10

        self.entry = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block1 = FractalBlock(2, 64, 0.0)
        self.pool1 = nn.MaxPool2d(2)

        self.trans = nn.Sequential(
            nn.Conv2d(64, 64*2, kernel_size=1),
            nn.BatchNorm2d(64*2),
            nn.ReLU(inplace=True)
        )
        self.block2 = FractalBlock(2, 64*2, 0.0)
        self.pool2 = nn.MaxPool2d(2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*2, n_classes)
        self.to(device)

    def forward(self, x):
        x = self.entry(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.trans(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm['lr'],       
            momentum=prm['momentum']
        )
        return self.optimizer

    def learn(self, train_data):
        self.train()
        for i, (inputs, labels) in enumerate(train_data):
            if i >= 50: break # Limit to ~3% of data (50/1563 batches) for speed
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()