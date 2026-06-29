import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Block 1
        self.block1 = ResidualBlock(in_channels=3, out_channels=192, kernel_size=3, stride=1, padding=1)  # output shape: (192, 32, 32)
        
        # Convolutional Block 2
        self.block2 = ResidualBlock(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)  # output shape: (192, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (192, 16, 16)
        
        # Convolutional Block 3
        self.block3 = ResidualBlock(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)  # output shape: (192, 16, 16)
        
        # Convolutional Block 4
        self.block4 = ResidualBlock(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)  # output shape: (192, 16, 16)
        
        # Attention Mechanism 1
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1), 
            nn.BatchNorm2d(num_features=192, momentum=0.9, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )  # output shape: (1, 16, 16)
        
        # Residual Connection 1
        self.residual1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)  # output shape: (192, 16, 16)
        
        # Convolutional Block 5
        self.block5 = ResidualBlock(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)  # output shape: (192, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (192, 8, 8)
        
        # Convolutional Block 6
        self.block6 = ResidualBlock(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)  # output shape: (192, 8, 8)
        
        # Attention Mechanism 2
        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1), 
            nn.BatchNorm2d(num_features=192, momentum=0.9, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )  # output shape: (1, 8, 8)
        
        # Residual Connection 2
        self.residual2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)  # output shape: (192, 8, 8)
        
        # Convolutional Block 7
        self.block7 = ResidualBlock(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)  # output shape: (192, 8, 8)
        
        # Adaptive Pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # output shape: (192, 1, 1)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1)  # output shape: (192)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=192, out_features=192)  # output shape: (192)
        self.bn5 = nn.BatchNorm1d(num_features=192, momentum=0.9, eps=1e-4)  # output shape: (192)
        
        # SiLU Activation
        self.silu5 = nn.SiLU()  # output shape: (192)
        
        # Dropout Regularization
        self.dropout1 = nn.Dropout(p=0.3)  # output shape: (192)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=192, out_features=10)  # output shape: (10)

    def forward(self, x):
        x = self.block1(x)  # output shape: (192, 32, 32)
        x = self.block2(x)  # output shape: (192, 32, 32)
        x = self.pool1(x)  # output shape: (192, 16, 16)
        x = self.block3(x)  # output shape: (192, 16, 16)
        x = self.block4(x)  # output shape: (192, 16, 16)
        
        att1 = self.attention1(x)
        x = att1 * x + self.residual1(x)  # output shape: (192, 16, 16)
        
        x = self.block5(x)  # output shape: (192, 16, 16)
        x = self.pool2(x)  # output shape: (192, 8, 8)
        x = self.block6(x)  # output shape: (192, 8, 8)
        
        att2 = self.attention2(x)
        x = att2 * x + self.residual2(x)  # output shape: (192, 8, 8)
        
        x = self.block7(x)  # output shape: (192, 8, 8)
        x = self.pool(x)  # output shape: (192, 1, 1)
        x = self.flatten(x)  # output shape: (192)
        x = self.silu5(self.bn5(self.fc1(x)))  # output shape: (192)
        x = self.dropout1(x)  # output shape: (192)
        x = self.fc2(x)  # output shape: (10)
        
        return x