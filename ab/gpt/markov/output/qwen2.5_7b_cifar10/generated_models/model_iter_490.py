class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=2, stride=1, padding=0) # output shape: (128, 31, 31)
        
        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(num_features=128, momentum=0.8, eps=1e-5) # output shape: (128, 31, 31)
        
        # SiLU Activation
        self.silu1 = nn.SiLU() # output shape: (128, 31, 31)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (128, 15, 15)
        
        # Attention Mechanism 1
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1), 
            nn.BatchNorm2d(num_features=128, momentum=0.8, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 15, 15)
        
        # Residual Connection 1
        self.residual1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1) # output shape: (128, 15, 15)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=0) # output shape: (256, 15, 15)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=256, momentum=0.8, eps=1e-5) # output shape: (256, 15, 15)
        
        # SiLU Activation
        self.silu2 = nn.SiLU() # output shape: (256, 15, 15)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (256, 7, 7)
        
        # Attention Mechanism 2
        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1), 
            nn.BatchNorm2d(num_features=256, momentum=0.8, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 7, 7)
        
        # Residual Connection 2
        self.residual2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1) # output shape: (256, 7, 7)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=0) # output shape: (512, 7, 7)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=512, momentum=0.8, eps=1e-5) # output shape: (512, 7, 7)
        
        # SiLU Activation
        self.silu3 = nn.SiLU() # output shape: (512, 7, 7)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (512, 3, 3)
        
        # Attention Mechanism 3
        self.attention3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1), 
            nn.BatchNorm2d(num_features=512, momentum=0.8, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 3, 3)
        
        # Residual Connection 3
        self.residual3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1) # output shape: (512, 3, 3)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1) # output shape: (512*3*3)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=512*3*3, out_features=256) # output shape: (256)
        
        # Batch Normalization 4
        self.bn4 = nn.BatchNorm1d(num_features=256, momentum=0.8, eps=1e-5) # output shape: (256)
        
        # SiLU Activation
        self.silu4 = nn.SiLU() # output shape: (256)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=256, out_features=10) # output shape: (10)

    def forward(self, x):
        # Apply attention mechanisms and max pooling
        x = self.pool1(self.silu1(self.bn1(self.conv1(x)))) # output shape: (128, 15, 15)
        att1 = self.attention1(x)
        x = att1 * x + self.residual1(x) # output shape: (128, 15, 15)
        
        x = self.pool2(self.silu2(self.bn2(self.conv2(x)))) # output shape: (256, 7, 7)
        att2 = self.attention2(x)
        x = att2 * x + self.residual2(x) # output shape: (256, 7, 7)
        
        x = self.pool3(self.silu3(self.bn3(self.conv3(x)))) # output shape: (512, 3, 3)
        att3 = self.attention3(x)
        x = att3 * x + self.residual3(x) # output shape: (512, 3, 3)
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x) # output shape: (512*3*3)
        x = self.silu4(self.bn4(self.fc1(x))) # output shape: (256)
        x = self.fc2(x) # output shape: (10)
        
        return x