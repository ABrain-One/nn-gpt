class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=3, stride=1, padding=1, bias=True) # output shape: (192, 32, 32)
        
        # Adaptive Normalization 1
        self.norm1 = nn.GroupNorm(num_groups=16, num_channels=192, eps=1e-5) # output shape: (192, 32, 32)
        
        # Swish Activation
        self.swish1 = nn.SiLU() # output shape: (192, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (192, 16, 16)
        
        # Attention Mechanism 1
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=True), 
            nn.GroupNorm(num_groups=16, num_channels=192, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, bias=True),
            nn.Sigmoid()
        ) # output shape: (1, 16, 16)
        
        # Residual Connection 1
        self.residual1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=True) # output shape: (192, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True) # output shape: (384, 16, 16)
        
        # Adaptive Normalization 2
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=384, eps=1e-5) # output shape: (384, 16, 16)
        
        # Swish Activation
        self.swish2 = nn.SiLU() # output shape: (384, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (384, 8, 8)
        
        # Attention Mechanism 2
        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, bias=True), 
            nn.GroupNorm(num_groups=32, num_channels=384, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=384, out_channels=1, kernel_size=1, bias=True),
            nn.Sigmoid()
        ) # output shape: (1, 8, 8)
        
        # Residual Connection 2
        self.residual2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, bias=True) # output shape: (384, 8, 8)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=768, kernel_size=3, stride=1, padding=1, bias=True) # output shape: (768, 8, 8)
        
        # Adaptive Normalization 3
        self.norm3 = nn.GroupNorm(num_groups=64, num_channels=768, eps=1e-5) # output shape: (768, 8, 8)
        
        # Swish Activation
        self.swish3 = nn.SiLU() # output shape: (768, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (768, 4, 4)
        
        # Attention Mechanism 3
        self.attention3 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=1, bias=True), 
            nn.GroupNorm(num_groups=64, num_channels=768, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, bias=True),
            nn.Sigmoid()
        ) # output shape: (1, 4, 4)
        
        # Residual Connection 3
        self.residual3 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=1, bias=True) # output shape: (768, 4, 4)
        
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=768, out_channels=1536, kernel_size=3, stride=1, padding=1, bias=True) # output shape: (1536, 4, 4)
        
        # Adaptive Normalization 4
        self.norm4 = nn.GroupNorm(num_groups=128, num_channels=1536, eps=1e-5) # output shape: (1536, 4, 4)
        
        # Swish Activation
        self.swish4 = nn.SiLU() # output shape: (1536, 4, 4)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (1536, 2, 2)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1) # output shape: (1536*2*2)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=1536*2*2, out_features=768, bias=True) # output shape: (768)
        
        # Adaptive Normalization 5
        self.norm5 = nn.LayerNorm(normalized_shape=768, eps=1e-5) # output shape: (768)
        
        # Swish Activation
        self.swish5 = nn.SiLU() # output shape: (768)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=768, out_features=10, bias=True) # output shape: (10)
    
    def forward(self, x):
        # Apply attention mechanisms and max pooling
        x = self.pool1(self.swish1(self.norm1(self.conv1(x)))) # output shape: (192, 16, 16)
        att1 = self.attention1(x)
        x = att1 * x + self.residual1(x) # output shape: (192, 16, 16)
        
        x = self.pool2(self.swish2(self.norm2(self.conv2(x)))) # output shape: (384, 8, 8)
        att2 = self.attention2(x)
        x = att2 * x + self.residual2(x) # output shape: (384, 8, 8)
        
        x = self.pool3(self.swish3(self.norm3(self.conv3(x)))) # output shape: (768, 4, 4)
        att3 = self.attention3(x)
        x = att3 * x + self.residual3(x) # output shape: (768, 4, 4)
        
        x = self.pool4(self.swish4(self.norm4(self.conv4(x)))) # output shape: (1536, 2, 2)
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x) # output shape: (1536*2*2)
        x = self.swish5(self.norm5(self.fc1(x))) # output shape: (768)
        x = self.fc2(x) # output shape: (10)
        
        return x