class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) # output shape: (32, 32, 32)
        
        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(num_features=32) # output shape: (32, 32, 32)
        
        # ReLU 1
        self.relu1 = nn.ReLU() # output shape: (32, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (32, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # output shape: (64, 16, 16)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=64) # output shape: (64, 16, 16)
        
        # ReLU 2
        self.relu2 = nn.ReLU() # output shape: (64, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (64, 8, 8)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # output shape: (128, 8, 8)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=128) # output shape: (128, 8, 8)
        
        # ReLU 3
        self.relu3 = nn.ReLU() # output shape: (128, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (128, 4, 4)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=512) # output shape: (512)
        
        # Dropout 1
        self.dropout1 = nn.Dropout(p=0.5) # output shape: (512)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=512, out_features=256) # output shape: (256)
        
        # Dropout 2
        self.dropout2 = nn.Dropout(p=0.5) # output shape: (256)
        
        # Fully Connected Layer 3
        self.fc3 = nn.Linear(in_features=256, out_features=10) # output shape: (10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x)))) # output shape: (32, 16, 16)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x)))) # output shape: (64, 8, 8)
        x = self.pool3(self.relu3(self.bn3(self.conv3(x)))) # output shape: (128, 4, 4)
        
        x = x.view(-1, 128*4*4) # output shape: (2048)
        x = self.dropout1(self.relu1(self.fc1(x))) # output shape: (512)
        x = self.dropout2(self.relu1(self.fc2(x))) # output shape: (256)
        x = self.fc3(x) # output shape: (10)
        return x