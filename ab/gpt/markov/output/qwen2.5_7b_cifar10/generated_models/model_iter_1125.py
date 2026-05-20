class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1) # output shape: (256, 32, 32)
        
        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(num_features=256, momentum=0.9, eps=1e-4) # output shape: (256, 32, 32)
        
        # SiLU Activation
        self.silu1 = nn.SiLU() # output shape: (256, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (256, 16, 16)
        
        # Attention Mechanism 1
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1), 
            nn.BatchNorm2d(num_features=256, momentum=0.9, eps=1e-4),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 16, 16)
        
        # Residual Connection 1
        self.residual1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1) # output shape: (256, 16, 16)
        
        # Dropout Layer 1
        self.dropout1 = nn.Dropout(p=0.3) # output shape: (256, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) # output shape: (512, 16, 16)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=512, momentum=0.9, eps=1e-4) # output shape: (512, 16, 16)
        
        # SiLU Activation
        self.silu2 = nn.SiLU() # output shape: (512, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (512, 8, 8)
        
        # Attention Mechanism 2
        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1), 
            nn.BatchNorm2d(num_features=512, momentum=0.9, eps=1e-4),
            nn.SiLU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 8, 8)
        
        # Residual Connection 2
        self.residual2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1) # output shape: (512, 8, 8)
        
        # Dropout Layer 2
        self.dropout2 = nn.Dropout(p=0.3) # output shape: (512, 8, 8)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1) # output shape: (1024, 8, 8)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=1024, momentum=0.9, eps=1e-4) # output shape: (1024, 8, 8)
        
        # SiLU Activation
        self.silu3 = nn.SiLU() # output shape: (1024, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (1024, 4, 4)
        
        # Attention Mechanism 3
        self.attention3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1), 
            nn.BatchNorm2d(num_features=1024, momentum=0.9, eps=1e-4),
            nn.SiLU(),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 4, 4)
        
        # Residual Connection 3
        self.residual3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1) # output shape: (1024, 4, 4)
        
        # Dropout Layer 3
        self.dropout3 = nn.Dropout(p=0.3) # output shape: (1024, 4, 4)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1) # output shape: (1024*4*4)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=1024*4*4, out_features=512, bias=True) # output shape: (512)
        
        # Batch Normalization 4
        self.bn4 = nn.BatchNorm1d(num_features=512, momentum=0.9, eps=1e-4) # output shape: (512)
        
        # SiLU Activation
        self.silu4 = nn.SiLU() # output shape: (512)
        
        # Dropout Layer 4
        self.dropout4 = nn.Dropout(p=0.3) # output shape: (512)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=True) # output shape: (10)

    def forward(self, x):
        # Ensure consistent input preprocessing
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        
        # Apply residual connections and attention mechanisms
        x = self.pool1(self.silu1(self.bn1(self.conv1(x)))) # output shape: (256, 16, 16)
        x = self.dropout1(self.silu1(self.bn1(self.conv1(x) + self.residual1(x) * self.attention1(x)))) # output shape: (256, 16, 16)
        
        x = self.pool2(self.silu2(self.bn2(self.conv2(x)))) # output shape: (512, 8, 8)
        x = self.dropout2(self.silu2(self.bn2(self.conv2(x) + self.residual2(x) * self.attention2(x)))) # output shape: (512, 8, 8)
        
        x = self.pool3(self.silu3(self.bn3(self.conv3(x)))) # output shape: (1024, 4, 4)
        x = self.dropout3(self.silu3(self.bn3(self.conv3(x) + self.residual3(x) * self.attention3(x)))) # output shape: (1024, 4, 4)
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x) # output shape: (1024*4*4)
        x = self.silu4(self.bn4(self.fc1(x))) # output shape: (512)
        x = self.dropout4(self.fc2(x)) # output shape: (10)
        
        return x