class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) # output shape: (32, 32, 32)
        
        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(num_features=32, momentum=0.9, eps=1e-5) # output shape: (32, 32, 32)
        
        # SiLU Activation
        self.silu1 = nn.SiLU() # output shape: (32, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (32, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # output shape: (64, 16, 16)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5) # output shape: (64, 16, 16)
        
        # SiLU Activation
        self.silu2 = nn.SiLU() # output shape: (64, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (64, 8, 8)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # output shape: (128, 8, 8)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=128, momentum=0.9, eps=1e-5) # output shape: (128, 8, 8)
        
        # SiLU Activation
        self.silu3 = nn.SiLU() # output shape: (128, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (128, 4, 4)
        
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # output shape: (256, 4, 4)
        
        # Batch Normalization 4
        self.bn4 = nn.BatchNorm2d(num_features=256, momentum=0.9, eps=1e-5) # output shape: (256, 4, 4)
        
        # SiLU Activation
        self.silu4 = nn.SiLU() # output shape: (256, 4, 4)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (256, 2, 2)
        
        # Adaptive Feature Map Aggregation
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # output shape: (256, 1, 1)
        self.flatten = nn.Flatten() # output shape: (256)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=256, out_features=256) # output shape: (256)
        
        # SiLU Activation
        self.silu1 = nn.SiLU() # output shape: (256)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=256, out_features=10) # output shape: (10)
        
        # Spatial Attention Mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1), 
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (256, 2, 2)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.pool1(self.silu1(self.bn1(self.conv1(x)))) # output shape: (32, 16, 16)
        x = self.pool2(self.silu2(self.bn2(self.conv2(x)))) # output shape: (64, 8, 8)
        x = self.pool3(self.silu3(self.bn3(self.conv3(x)))) # output shape: (128, 4, 4)
        x = self.pool4(self.silu4(self.bn4(self.conv4(x)))) # output shape: (256, 2, 2)
        
        # Adaptive Feature Map Aggregation
        x = self.adaptive_avg_pool(x).view(-1, 256) # output shape: (256)
        
        # Apply spatial attention mechanism
        att = self.spatial_attention(x.unsqueeze(2).unsqueeze(3)) # output shape: (256, 2, 2, 1)
        att = att.squeeze(3).squeeze(2) # output shape: (256)
        x = att * x + x # output shape: (256)
        
        # Fully Connected Layers
        x = self.silu1(self.fc1(x)) # output shape: (256)
        x = self.fc2(x) # output shape: (10)
        
        return x