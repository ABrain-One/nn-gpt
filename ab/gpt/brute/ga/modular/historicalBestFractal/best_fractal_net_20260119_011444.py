
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class FractalDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.15):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # If testing, Average everything
        if not self.training: 
            return torch.stack(inputs).mean(dim=0)
        
        n = len(inputs)
        # Global Drop Path Logic
        mask = torch.bernoulli(torch.full((n,), 1 - self.drop_prob, device=inputs[0].device))
        
        # Ensure at least one path is active
        if mask.sum() == 0: 
            mask[torch.randint(0, n, (1,)).item()] = 1.0
            
        active = [inp for inp, m in zip(inputs, mask) if m > 0]
        return torch.stack(active).mean(dim=0)

class FractalBlock(nn.Module):
    def __init__(self, n_columns: int, channels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.n_columns = n_columns
        
        # Base computation: Conv -> BN -> ReLU
        # Crucial: Keeps channels same (C -> C) so we can stack recursively
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )





        if n_columns > 1:
            self.left = FractalBlock(n_columns - 1, channels, dropout_prob)
            self.right_1 = FractalBlock(n_columns - 1, channels, dropout_prob)
            self.right_2 = FractalBlock(n_columns - 1, channels, dropout_prob)
            self.join = FractalDropPath(drop_prob=dropout_prob)

    def forward(self, x):
        if self.n_columns == 1: 
            return self.conv(x)
        
        # Fractal Structure:
        # Left: Shallow path
        # Right: Deep path (Recursive stack)
        out_left = self.left(x)
        out_right = self.right_2(self.right_1(x))
        
        return self.join([out_left, out_right])

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super(Net, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.prm = prm
        self.device = device
        
        c_in = in_shape[0] # Fix: [0] is channels (1 for MNIST), [1] is height
        n_classes = out_shape[0]
        
        # 1. Entry Block (Increase channels)
        self.entry = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 2. Fractal Stage 1 (32 channels)
        self.block1 = FractalBlock(n_columns=2, channels=32, dropout_prob=0.1)
        self.pool1 = nn.MaxPool2d(2)
        
        # 3. Transition (32 -> 64)
        self.trans = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 4. Fractal Stage 2 (64 channels)
        self.block2 = FractalBlock(n_columns=2, channels=64, dropout_prob=0.1)
        self.pool2 = nn.MaxPool2d(2)
        
        # 5. Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, n_classes)
        
        self.to(device)

    def forward(self, x):
        x = self.entry(x)       # 1 -> 32
        x = self.block1(x)      # 32 -> 32
        x = self.pool1(x)       # Downsample
        
        x = self.trans(x)       # 32 -> 64
        x = self.block2(x)      # 64 -> 64
        x = self.pool2(x)       # Downsample
        
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=prm.get('lr', 00.01), 
            betas=(prm.get('beta1', 0.9), prm.get('beta2', 0.999)), 
            eps=prm.get('eps', 1e-8), 
            weight_decay=prm.get('weight_decay', 0), 
            amsgrad=prm.get('amsgrad', False)
        )

        return self.optimizer

    def learn(self, train_data):
        self.train()
        total_loss = 0
        count = 0
        
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
        # Return average loss so API knows we are alive
        return total_loss / count if count > 0 else 0.0
