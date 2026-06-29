class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # output shape: (64, 32, 32)
        
        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5) # output shape: (64, 32, 32)
        
        # SiLU Activation
        self.silu1 = nn.SiLU() # output shape: (64, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (64, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # output shape: (128, 16, 16)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=128, momentum=0.9, eps=1e-5) # output shape: (128, 16, 16)
        
        # SiLU Activation
        self.silu2 = nn.SiLU() # output shape: (128, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (128, 8, 8)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # output shape: (256, 8, 8)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=256, momentum=0.9, eps=1e-5) # output shape: (256, 8, 8)
        
        # SiLU Activation
        self.silu3 = nn.SiLU() # output shape: (256, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (256, 4, 4)
        
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) # output shape: (512, 4, 4)
        
        # Batch Normalization 4
        self.bn4 = nn.BatchNorm2d(num_features=512, momentum=0.9, eps=1e-5) # output shape: (512, 4, 4)
        
        # SiLU Activation
        self.silu4 = nn.SiLU() # output shape: (512, 4, 4)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (512, 2, 2)
        
        # Feature Pyramid Network (FPN)
        self.lateral_conv1 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=1) # output shape: (512, 16, 16)
        self.lateral_conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1) # output shape: (512, 8, 8)
        self.lateral_conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1) # output shape: (512, 4, 4)
        
        self.fpn_up1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2) # output shape: (512, 32, 32)
        self.fpn_up2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=4) # output shape: (512, 64, 64)
        
        # Adaptive Feature Map Aggregation
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # output shape: (512, 1, 1)
        self.flatten = nn.Flatten() # output shape: (512)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=512, out_features=512) # output shape: (512)
        self.bn1 = nn.BatchNorm1d(num_features=512, momentum=0.9, eps=1e-5) # output shape: (512)
        self.silu1 = nn.SiLU() # output shape: (512)
        self.dropout1 = nn.Dropout(p=0.4) # output shape: (512)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=512, out_features=10) # output shape: (10)
        
        # Spatial Attention Mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1), 
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (512, 2, 2)
    
    def forward(self, x):
        # Forward pass through convolutional layers
        x1 = self.pool1(self.silu1(self.bn1(self.conv1(x)))) # output shape: (64, 16, 16)
        x2 = self.pool2(self.silu2(self.bn2(self.conv2(x1)))) # output shape: (128, 8, 8)
        x3 = self.pool3(self.silu3(self.bn3(self.conv3(x2)))) # output shape: (256, 4, 4)
        x4 = self.pool4(self.silu4(self.bn4(self.conv4(x3)))) # output shape: (512, 2, 2)
        
        # Feature Pyramid Network (FPN)
        p1 = self.lateral_conv1(x1) # output shape: (512, 16, 16)
        p2 = self.lateral_conv2(x2) # output shape: (512, 8, 8)
        p3 = self.lateral_conv3(x3) # output shape: (512, 4, 4)
        
        p2 = p2 + self.fpn_up1(p1) # output shape: (512, 32, 32)
        p3 = p3 + self.fpn_up2(p2) # output shape: (512, 64, 64)
        
        # Adaptive Feature Map Aggregation
        p1 = self.adaptive_avg_pool(p1).view(-1, 512) # output shape: (512)
        p2 = self.adaptive_avg_pool(p2).view(-1, 512) # output shape: (512)
        p3 = self.adaptive_avg_pool(p3).view(-1, 512) # output shape: (512)
        p4 = self.adaptive_avg_pool(x4).view(-1, 512) # output shape: (512)
        
        x = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1), p3.unsqueeze(1), p4.unsqueeze(1)], dim=1) # output shape: (batch_size, 4, 512)
        x = x.mean(dim=1) # output shape: (batch_size, 512)
        
        # Apply spatial attention mechanism
        x = x * self.spatial_attention(x1.unsqueeze(1).unsqueeze(1)).squeeze(3).squeeze(2) + x
        
        # Fully Connected Layers
        x = self.silu1(self.bn1(self.fc1(x))) # output shape: (512)
        x = self.dropout1(x)
        x = self.fc2(x) # output shape: (10)
        
        return x