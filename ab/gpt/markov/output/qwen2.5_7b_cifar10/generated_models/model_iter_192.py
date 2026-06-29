class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Depthwise Separable Convolutional Layer 1
        self.dsc1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, groups=3), 
            nn.BatchNorm2d(num_features=64, momentum=0.8, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64, momentum=0.8, eps=1e-5),
            nn.SiLU()
        ) # output shape: (64, 32, 32)
        
        # Spatial Pyramid Pooling 1
        self.spp1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(8, 8)), # output shape: (64, 8, 8)
            nn.AdaptiveAvgPool2d(output_size=(4, 4)), # output shape: (64, 4, 4)
            nn.AdaptiveAvgPool2d(output_size=(2, 2))  # output shape: (64, 2, 2)
        )
        
        # Attention Mechanism 1
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), 
            nn.BatchNorm2d(num_features=64, momentum=0.8, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(16, 16)) # output shape: (64, 16, 16)
        
        # Depthwise Separable Convolutional Layer 2
        self.dsc2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, groups=64), 
            nn.BatchNorm2d(num_features=128, momentum=0.8, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128, momentum=0.8, eps=1e-5),
            nn.SiLU()
        ) # output shape: (128, 16, 16)
        
        # Spatial Pyramid Pooling 2
        self.spp2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(8, 8)), # output shape: (128, 8, 8)
            nn.AdaptiveAvgPool2d(output_size=(4, 4)), # output shape: (128, 4, 4)
            nn.AdaptiveAvgPool2d(output_size=(2, 2))  # output shape: (128, 2, 2)
        )
        
        # Attention Mechanism 2
        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1), 
            nn.BatchNorm2d(num_features=128, momentum=0.8, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(8, 8)) # output shape: (128, 8, 8)
        
        # Depthwise Separable Convolutional Layer 3
        self.dsc3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, groups=128), 
            nn.BatchNorm2d(num_features=256, momentum=0.8, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(num_features=256, momentum=0.8, eps=1e-5),
            nn.SiLU()
        ) # output shape: (256, 8, 8)
        
        # Spatial Pyramid Pooling 3
        self.spp3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(4, 4)), # output shape: (256, 4, 4)
            nn.AdaptiveAvgPool2d(output_size=(2, 2))  # output shape: (256, 2, 2)
        )
        
        # Attention Mechanism 3
        self.attention3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1), 
            nn.BatchNorm2d(num_features=256, momentum=0.8, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.AdaptiveAvgPool2d(output_size=(4, 4)) # output shape: (256, 4, 4)
        
        # Depthwise Separable Convolutional Layer 4
        self.dsc4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, groups=256), 
            nn.BatchNorm2d(num_features=512, momentum=0.8, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(num_features=512, momentum=0.8, eps=1e-5),
            nn.SiLU()
        ) # output shape: (512, 4, 4)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1) # output shape: (512*4*4)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=512*4*4, out_features=1024) # output shape: (1024)
        
        # Batch Normalization 5
        self.bn5 = nn.BatchNorm1d(num_features=1024, momentum=0.8, eps=1e-5) # output shape: (1024)
        
        # SiLU Activation
        self.silu5 = nn.SiLU() # output shape: (1024)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=1024, out_features=10) # output shape: (10)
    
    def forward(self, x):
        # Apply attention mechanisms and max pooling
        x = self.pool1(self.dsc1(x)) # output shape: (64, 16, 16)
        spp1 = self.spp1(x)
        spp1 = torch.cat([spp1[0], spp1[1], spp1[2]], dim=1) # output shape: (192, 16, 16)
        att1 = self.attention1(spp1)
        x = att1 * x + x # output shape: (64, 16, 16)
        
        x = self.pool2(self.dsc2(x)) # output shape: (128, 8, 8)
        spp2 = self.spp2(x)
        spp2 = torch.cat([spp2[0], spp2[1], spp2[2]], dim=1) # output shape: (384, 8, 8)
        att2 = self.attention2(spp2)
        x = att2 * x + x # output shape: (128, 8, 8)
        
        x = self.pool3(self.dsc3(x)) # output shape: (256, 4, 4)
        spp3 = self.spp3(x)
        spp3 = torch.cat([spp3[0], spp3[1]], dim=1) # output shape: (512, 4, 4)
        att3 = self.attention3(spp3)
        x = att3 * x + x # output shape: (256, 4, 4)
        
        x = self.dsc4(x) # output shape: (512