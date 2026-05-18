class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) # output shape: (32, 32, 32)
        
        # LayerNorm 1
        self.ln1 = nn.LayerNorm(normalized_shape=(32, 32, 32)) # output shape: (32, 32, 32)
        
        # SiLU Activation
        self.silu1 = nn.SiLU() # output shape: (32, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (32, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # output shape: (64, 16, 16)
        
        # LayerNorm 2
        self.ln2 = nn.LayerNorm(normalized_shape=(64, 16, 16)) # output shape: (64, 16, 16)
        
        # SiLU Activation
        self.silu2 = nn.SiLU() # output shape: (64, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (64, 8, 8)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # output shape: (128, 8, 8)
        
        # LayerNorm 3
        self.ln3 = nn.LayerNorm(normalized_shape=(128, 8, 8)) # output shape: (128, 8, 8)
        
        # SiLU Activation
        self.silu3 = nn.SiLU() # output shape: (128, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (128, 4, 4)
        
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # output shape: (256, 4, 4)
        
        # LayerNorm 4
        self.ln4 = nn.LayerNorm(normalized_shape=(256, 4, 4)) # output shape: (256, 4, 4)
        
        # SiLU Activation
        self.silu4 = nn.SiLU() # output shape: (256, 4, 4)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (256, 2, 2)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=256*2*2, out_features=1024) # output shape: (1024)
        
        # LayerNorm 5
        self.ln5 = nn.LayerNorm(normalized_shape=(1024,)) # output shape: (1024)
        
        # SiLU Activation
        self.silu5 = nn.SiLU() # output shape: (1024)
        
        # Dropout 1
        self.dropout1 = nn.Dropout(p=0.3) # output shape: (1024)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=1024, out_features=10) # output shape: (10)

    def forward(self, x):
        x = self.pool1(self.silu1(self.ln1(self.conv1(x)))) # output shape: (32, 16, 16)
        x = self.pool2(self.silu2(self.ln2(self.conv2(x)))) # output shape: (64, 8, 8)
        x = self.pool3(self.silu3(self.ln3(self.conv3(x)))) # output shape: (128, 4, 4)
        x = self.pool4(self.silu4(self.ln4(self.conv4(x)))) # output shape: (256, 2, 2)
        
        x = x.view(-1, 256*2*2) # output shape: (1024)
        x = self.silu5(self.ln5(self.fc1(x))) # output shape: (1024)
        x = self.dropout1(x) # output shape: (1024)
        x = self.fc2(x) # output shape: (10)
        
        return x