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
        self.block5 = ResidualBlock(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)  # output shape: (384, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (384, 8, 8)
        
        # Convolutional Block 6
        self.block6 = ResidualBlock(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)  # output shape: (384, 8, 8)
        
        # Convolutional Block 7
        self.block7 = ResidualBlock(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)  # output shape: (384, 8, 8)
        
        # Attention Mechanism 2
        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1), 
            nn.BatchNorm2d(num_features=384, momentum=0.9, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )  # output shape: (1, 8, 8)
        
        # Residual Connection 2
        self.residual2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1)  # output shape: (384, 8, 8)
        
        # Convolutional Block 8
        self.block8 = ResidualBlock(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)  # output shape: (384, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (384, 4, 4)
        
        # Convolutional Block 9
        self.block9 = ResidualBlock(in_channels=384, out_channels=768, kernel_size=3, stride=1, padding=1)  # output shape: (768, 4, 4)
        
        # Convolutional Block 10
        self.block10 = ResidualBlock(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)  # output shape: (768, 4, 4)
        
        # Attention Mechanism 3
        self.attention3 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=1), 
            nn.BatchNorm2d(num_features=768, momentum=0.9, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )  # output shape: (1, 4, 4)
        
        # Residual Connection 3
        self.residual3 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=1)  # output shape: (768, 4, 4)
        
        # Convolutional Block 11
        self.block11 = ResidualBlock(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)  # output shape: (768, 4, 4)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (768, 2, 2)
        
        # Convolutional Block 12
        self.block12 = ResidualBlock(in_channels=768, out_channels=1536, kernel_size=3, stride=1, padding=1)  # output shape: (1536, 2, 2)
        
        # Convolutional Block 13
        self.block13 = ResidualBlock(in_channels=1536, out_channels=1536, kernel_size=3, stride=1, padding=1)  # output shape: (1536, 2, 2)
        
        # Adaptive Pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # output shape: (1536, 1, 1)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1)  # output shape: (1536)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=1536, out_features=768)  # output shape: (768)
        self.bn5 = nn.BatchNorm1d(num_features=768, momentum=0.9, eps=1e-4)  # output shape: (768)
        
        # SiLU Activation
        self.silu5 = nn.SiLU()  # output shape: (768)
        
        # Dropout Regularization
        self.dropout1 = nn.Dropout(p=0.3)  # output shape: (768)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=768, out_features=10)  # output shape: (10)

    def forward(self, x):
        # Process through the first two blocks
        x = self.block1(x)  # output shape: (192, 32, 32)
        x = self.block2(x)  # output shape: (192, 32, 32)
        x = self.pool1(x)   # output shape: (192, 16, 16)
        
        # Process through the next two blocks
        x = self.block3(x)  # output shape: (192, 16, 16)
        x = self.block4(x)  # output shape: (192, 16, 16)
        
        # Apply attention mechanism 1 and residual connection
        att1 = self.attention1(x)
        x = att1 * x + self.residual1(x)  # output shape: (192, 16, 16)
        
        # Process through the next three blocks
        x = self.block5(x)  # output shape: (384, 16, 16)
        x = self.block6(x)  # output shape: (38