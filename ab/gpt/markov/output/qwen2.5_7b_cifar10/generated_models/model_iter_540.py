import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=3, stride=1, padding=1) # output shape: (192, 32, 32)
        
        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(num_features=192, momentum=0.9, eps=1e-5) # output shape: (192, 32, 32)
        
        # SiLU Activation
        self.silu1 = nn.SiLU() # output shape: (192, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (192, 16, 16)
        
        # Attention Mechanism 1
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1), 
            nn.BatchNorm2d(num_features=192, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 16, 16)
        
        # Residual Connection 1
        self.residual1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1) # output shape: (192, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=2, stride=1, padding=0) # output shape: (384, 15, 15)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=384, momentum=0.9, eps=1e-5) # output shape: (384, 15, 15)
        
        # SiLU Activation
        self.silu2 = nn.SiLU() # output shape: (384, 15, 15)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (384, 7, 7)
        
        # Attention Mechanism 2
        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1), 
            nn.BatchNorm2d(num_features=384, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=384, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 7, 7)
        
        # Residual Connection 2
        self.residual2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1) # output shape: (384, 7, 7)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=768, kernel_size=2, stride=1, padding=0) # output shape: (768, 6, 6)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=768, momentum=0.9, eps=1e-5) # output shape: (768, 6, 6)
        
        # SiLU Activation
        self.silu3 = nn.SiLU() # output shape: (768, 6, 6)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (768, 3, 3)
        
        # Attention Mechanism 3
        self.attention3 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=1), 
            nn.BatchNorm2d(num_features=768, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 3, 3)
        
        # Residual Connection 3
        self.residual3 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=1) # output shape: (768, 3, 3)
        
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=768, out_channels=1536, kernel_size=2, stride=1, padding=0) # output shape: (1536, 2, 2)
        
        # Batch Normalization 4
        self.bn4 = nn.BatchNorm2d(num_features=1536, momentum=0.9, eps=1e-5) # output shape: (1536, 2, 2)
        
        # SiLU Activation
        self.silu4 = nn.SiLU() # output shape: (1536, 2, 2)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (1536, 1, 1)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1) # output shape: (1536)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=1536, out_features=768) # output shape: (768)
        
        # Batch Normalization 5
        self.bn5 = nn.BatchNorm1d(num_features=768, momentum=0.9, eps=1e-5) # output shape: (768)
        
        # SiLU Activation
        self.silu5 = nn.SiLU() # output shape: (768)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=768, out_features=10) # output shape: (10)
        
        # Skip Connection from Input to Second Conv Layer
        self.skip_connection = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=1) # output shape: (384, 16, 16)

    def forward(self, x):
        # Apply attention mechanisms and max pooling
        x = self.pool1(self.silu1(self.bn1(self.conv1(x)))) # output shape: (192, 16, 16)
        att1 = self.attention1(x)
        x = att1 * x + self.residual1(x) # output shape: (192, 16, 16)
        
        # Apply skip connection
        skip_x = self.skip_connection(x) # output shape: (384, 16, 16)
        x = torch.cat([skip_x, x], dim=1) # output shape: (576, 16, 16)
        
        x = self.pool2(self.silu2(self.bn2(self.conv2(x)))) # output shape: (384, 7, 7)
        att2 = self.attention2(x)
        x = att2 * x + self.residual2(x) # output shape: (384, 7, 7)
        
        x = self.pool3(self.silu3(self.bn3(self.conv3(x)))) # output shape: (768, 3, 3)
        att3 = self.attention3(x)
        x = att3 * x + self.residual3(x) # output shape: (768, 3, 3)
        
        x = self.pool4(self.silu4(self.bn4(self.conv4(x)))) # output shape: (1536, 1, 1)
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x) # output shape: (1536)
        x = self.silu5(self.bn5(self.fc1(x))) # output shape