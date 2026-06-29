class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=3, stride=1, padding=1) # output shape: (192, 32, 32)
        
        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(num_features=192, momentum=0.9, eps=1e-4) # output shape: (192, 32, 32)
        
        # SiLU Activation
        self.silu1 = nn.SiLU() # output shape: (192, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (192, 16, 16)
        
        # Squeeze-and-Excitation Block 1
        self.se_block1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # output shape: (192, 1, 1)
            nn.Conv2d(in_channels=192, out_channels=192 // 4, kernel_size=1), # output shape: (48, 1, 1)
            nn.ReLU(), # output shape: (48, 1, 1)
            nn.Conv2d(in_channels=192 // 4, out_channels=192, kernel_size=1), # output shape: (192, 1, 1)
            nn.Sigmoid() # output shape: (192, 1, 1)
        )
        self.fc_se1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1) # output shape: (192, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1) # output shape: (384, 16, 16)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=384, momentum=0.9, eps=1e-4) # output shape: (384, 16, 16)
        
        # SiLU Activation
        self.silu2 = nn.SiLU() # output shape: (384, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (384, 8, 8)
        
        # Squeeze-and-Excitation Block 2
        self.se_block2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # output shape: (384, 1, 1)
            nn.Conv2d(in_channels=384, out_channels=384 // 4, kernel_size=1), # output shape: (96, 1, 1)
            nn.ReLU(), # output shape: (96, 1, 1)
            nn.Conv2d(in_channels=384 // 4, out_channels=384, kernel_size=1), # output shape: (384, 1, 1)
            nn.Sigmoid() # output shape: (384, 1, 1)
        )
        self.fc_se2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1) # output shape: (384, 8, 8)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=768, kernel_size=3, stride=1, padding=1) # output shape: (768, 8, 8)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=768, momentum=0.9, eps=1e-4) # output shape: (768, 8, 8)
        
        # SiLU Activation
        self.silu3 = nn.SiLU() # output shape: (768, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (768, 4, 4)
        
        # Squeeze-and-Excitation Block 3
        self.se_block3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # output shape: (768, 1, 1)
            nn.Conv2d(in_channels=768, out_channels=768 // 4, kernel_size=1), # output shape: (192, 1, 1)
            nn.ReLU(), # output shape: (192, 1, 1)
            nn.Conv2d(in_channels=768 // 4, out_channels=768, kernel_size=1), # output shape: (768, 1, 1)
            nn.Sigmoid() # output shape: (768, 1, 1)
        )
        self.fc_se3 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=1) # output shape: (768, 4, 4)
        
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=768, out_channels=1536, kernel_size=3, stride=1, padding=1) # output shape: (1536, 4, 4)
        
        # Batch Normalization 4
        self.bn4 = nn.BatchNorm2d(num_features=1536, momentum=0.9, eps=1e-4) # output shape: (1536, 4, 4)
        
        # SiLU Activation
        self.silu4 = nn.SiLU() # output shape: (1536, 4, 4)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (1536, 2, 2)
        
        # Squeeze-and-Excitation Block 4
        self.se_block4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # output shape: (1536, 1, 1)
            nn.Conv2d(in_channels=1536, out_channels=1536 // 4, kernel_size=1), # output shape: (384, 1, 1)
            nn.ReLU(), # output shape: (384, 1, 1)
            nn.Conv2d(in_channels=1536 // 4, out_channels=1536, kernel_size=1), # output shape: (1536, 1, 1)
            nn.Sigmoid() # output shape: (1536, 1, 1)
        )
        self.fc_se4 = nn.Conv2d(in_channels=1536, out_channels=1536, kernel_size=1) # output shape: (1536, 2, 2)
        
        # Final Processing Block
        self.final_block = nn.Sequential(
            nn.BatchNorm2d(num_features=1536, momentum=0.9, eps=1e-4), # output shape: (1536, 2, 2)
            nn.SiLU() # output shape: (1536, 2, 2)
        )
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1) # output shape: (1536*2*2)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=1536*2*2, out_features=768) # output shape: (768)
        
        # Batch Normalization 5
        self.bn5 = nn.BatchNorm1d(num_features=768, momentum=0.9, eps=1e-4) # output shape: (768)
        
        # SiLU Activation
        self.silu5 = nn.SiLU() # output shape: (768)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=768, out_features=10) # output shape: (10)

    def forward(self, x):
        # Apply attention mechanisms and max pooling
        x = self.pool1(self.silu1(self.bn1(self.conv1(x)))) # output shape: (192, 16, 16)
        se1 = self.se_block1(x)
        x = x * se1 + self.fc_se1(x) # output shape: