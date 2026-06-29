class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Block 1
        self.block1 = ResidualBlock(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)  # output shape: (128, 32, 32)
        
        # Convolutional Block 2
        self.block2 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # output shape: (128, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (128, 16, 16)
        
        # Convolutional Block 3
        self.block3 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # output shape: (128, 16, 16)
        
        # Convolutional Block 4
        self.block4 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # output shape: (128, 16, 16)
        
        # Attention Mechanism 1
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1), 
            nn.BatchNorm2d(num_features=128, momentum=0.9, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )  # output shape: (1, 16, 16)
        
        # Residual Connection 1
        self.residual1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)  # output shape: (128, 16, 16)
        
        # Convolutional Block 5
        self.block5 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # output shape: (128, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (128, 8, 8)
        
        # Convolutional Block 6
        self.block6 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # output shape: (128, 8, 8)
        
        # Attention Mechanism 2
        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1), 
            nn.BatchNorm2d(num_features=128, momentum=0.9, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )  # output shape: (1, 8, 8)
        
        # Residual Connection 2
        self.residual2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)  # output shape: (128, 8, 8)
        
        # Convolutional Block 7
        self.block7 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # output shape: (128, 8, 8)
        
        # Adaptive Pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # output shape: (128, 1, 1)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1)  # output shape: (128)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=128, out_features=128)  # output shape: (128)
        self.bn5 = nn.BatchNorm1d(num_features=128, momentum=0.9, eps=1e-4)  # output shape: (128)
        
        # SiLU Activation
        self.silu5 = nn.SiLU()  # output shape: (128)
        
        # Dropout Regularization
        self.dropout1 = nn.Dropout(p=0.3)  # output shape: (128)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # output shape: (10)

    def forward(self, x):
        x = self.block1(x)  # output shape: (128, 32, 32)
        x = self.block2(x)  # output shape: (128, 32, 32)
        x = self.pool1(x)  # output shape: (128, 16, 16)
        x = self.block3(x)  # output shape: (128, 16, 16)
        x = self.block4(x)  # output shape: (128, 16, 16)
        
        att1 = self.attention1(x)
        x = att1 * x + self.residual1(x)  # output shape: (128, 16, 16)
        
        x = self.block5(x)  # output shape: (128, 16, 16)
        x = self.pool2(x)  # output shape: (128, 8, 8)
        x = self.block6(x)  # output shape: (128, 8, 8)
        
        att2 = self.attention2(x)
        x = att2 * x + self.residual2(x)  # output shape: (128, 8, 8)
        
        x = self.block7(x)  # output shape: (128, 8, 8)
        x = self.pool(x)  # output shape: (128, 1, 1)
        x = self.flatten(x)  # output shape: (128)
        x = self.silu5(self.bn5(self.fc1(x)))  # output shape: (128)
        x = self.dropout1(x)  # output shape: (128)
        x = self.fc2(x)  # output shape: (10)
        
        return x