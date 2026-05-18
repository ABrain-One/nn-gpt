class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Depthwise Separable Convolutional Layer 1
        self.dsc1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, groups=3), 
            nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5),
            nn.SiLU()
        ) # output shape: (64, 32, 32)
        
        # Residual Connection 1
        self.res1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (64, 16, 16)
        
        # Attention Mechanism 1
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), 
            nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 16, 16)
        
        # Depthwise Separable Convolutional Layer 2
        self.dsc2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, groups=64), 
            nn.BatchNorm2d(num_features=128, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128, momentum=0.9, eps=1e-5),
            nn.SiLU()
        ) # output shape: (128, 16, 16)
        
        # Residual Connection 2
        self.res2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (128, 8, 8)
        
        # Attention Mechanism 2
        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1), 
            nn.BatchNorm2d(num_features=128, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 8, 8)
        
        # Depthwise Separable Convolutional Layer 3
        self.dsc3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, groups=128), 
            nn.BatchNorm2d(num_features=256, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(num_features=256, momentum=0.9, eps=1e-5),
            nn.SiLU()
        ) # output shape: (256, 8, 8)
        
        # Residual Connection 3
        self.res3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (256, 4, 4)
        
        # Attention Mechanism 3
        self.attention3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1), 
            nn.BatchNorm2d(num_features=256, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 4, 4)
        
        # Depthwise Separable Convolutional Layer 4
        self.dsc4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, groups=256), 
            nn.BatchNorm2d(num_features=512, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(num_features=512, momentum=0.9, eps=1e-5),
            nn.SiLU()
        ) # output shape: (512, 4, 4)
        
        # Residual Connection 4
        self.res4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (512, 2, 2)
        
        # Attention Mechanism 4
        self.attention4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1), 
            nn.BatchNorm2d(num_features=512, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 2, 2)
        
        # Adaptive Average Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # output shape: (512, 1, 1)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1) # output shape: (512)
        
        # Batch Normalization Before Flatten
        self.bn6 = nn.BatchNorm1d(num_features=512, momentum=0.9, eps=1e-5) # output shape: (512)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=512, out_features=1024) # output shape: (1024)
        
        # Batch Normalization 5
        self.bn5 = nn.BatchNorm1d(num_features=1024, momentum=0.9, eps=1e-5) # output shape: (1024)
        
        # SiLU Activation
        self.silu5 = nn.SiLU() # output shape: (1024)
        
        # Dropout Layer 1
        self.dropout1 = nn.Dropout(p=0.3) # output shape: (1024)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=1024, out_features=10) # output shape: (10)
    
    def forward(self, x):
        # Apply attention mechanisms and max pooling
        x = self.pool1(self.dsc1(x)) + self.res1(x) # output shape: (64, 16, 16)
        att1 = self.attention1(x)
        x = att1 * x + x # output shape: (64, 16, 16)
        
        x = self.pool2(self.dsc2(x)) + self.res2(x) # output shape: (128, 8, 8)
        att2 = self.attention2(x)
        x = att2 * x + x # output shape: (128, 8, 8)
        
        x = self.pool3(self.dsc3(x)) + self.res3(x) # output shape: (256, 4, 4)
        att3 = self.attention3(x)
        x = att3 * x + x # output shape: (256, 4, 4)
        
        x = self.pool4(self.dsc4(x)) + self.res4(x) # output shape: (512, 2, 2)
        att4 = self.attention4(x)
        x = att4 * x + x # output shape: (512,