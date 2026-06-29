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
        self.fc1 = nn.Linear(in_features=256*2*2, out_features=512) # output shape: (512)
        self.bn1 = nn.BatchNorm1d(num_features=512, momentum=0.9, eps=1e-5) # output shape: (512)
        self.silu1 = nn.SiLU() # output shape: (512)
        
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # output shape: (512, 1, 1)
        
        # Multi-Head Self-Attention
        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1) # output shape: (512)
        
        # Fully Connected Layer 1
        self.fc2 = nn.Linear(in_features=512, out_features=256) # output shape: (256)
        self.bn2 = nn.BatchNorm1d(num_features=256, momentum=0.9, eps=1e-5) # output shape: (256)
        self.silu2 = nn.SiLU() # output shape: (256)
        self.dropout2 = nn.Dropout(p=0.5) # output shape: (256)
        
        # Fully Connected Layer 2
        self.fc3 = nn.Linear(in_features=256, out_features=128) # output shape: (128)
        self.bn3 = nn.BatchNorm1d(num_features=128, momentum=0.9, eps=1e-5) # output shape: (128)
        self.silu3 = nn.SiLU() # output shape: (128)
        self.dropout3 = nn.Dropout(p=0.5) # output shape: (128)
        
        # Fully Connected Layer 3
        self.fc4 = nn.Linear(in_features=128, out_features=10) # output shape: (10)
    
    def forward(self, x):
        # Forward pass through convolutional layers
        x1 = self.pool1(self.silu1(self.bn1(self.conv1(x)))) # output shape: (32, 16, 16)
        x2 = self.pool2(self.silu2(self.bn2(self.conv2(x1)))) # output shape: (64, 8, 8)
        x3 = self.pool3(self.silu3(self.bn3(self.conv3(x2)))) # output shape: (128, 4, 4)
        x4 = self.pool4(self.silu4(self.bn4(self.conv4(x3)))) # output shape: (256, 2, 2)
        
        # Adaptive Feature Map Aggregation
        x1 = self.avg_pool(x1).view(-1, 512) # output shape: (512)
        x2 = self.avg_pool(x2).view(-1, 512) # output shape: (512)
        x3 = self.avg_pool(x3).view(-1, 512) # output shape: (512)
        x4 = self.avg_pool(x4).view(-1, 512) # output shape: (512)
        
        # Concatenate feature maps and apply global average pooling
        x = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)], dim=1) # output shape: (batch_size, 4, 512)
        x, _ = self.mha(x, x, x) # output shape: (batch_size, 4, 512)
        x = x.mean(dim=1) # output shape: (batch_size, 512)
        
        # Fully Connected Layers
        x = self.silu1(self.bn1(self.fc1(x))) # output shape: (512)
        x = self.dropout2(x) # output shape: (512)
        x = self.silu2(self.bn2(self.fc2(x))) # output shape: (256)
        x = self.dropout3(x) # output shape: (256)
        x = self.silu3(self.bn3(self.fc3(x))) # output shape: (128)
        x = self.fc4(x) # output shape: (10)
        
        return x