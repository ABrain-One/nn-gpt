class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=3, stride=1, padding=1)  # output shape: (192, 32, 32)
        
        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(num_features=192, momentum=0.9, eps=1e-4)  # output shape: (192, 32, 32)
        
        # SiLU Activation
        self.silu1 = nn.SiLU()  # output shape: (192, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (192, 16, 16)
        
        # Residual Connection 1
        self.residual1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)  # output shape: (192, 16, 16)
        
        # Custom Self-Attention Mechanism 1
        self.self_attention1 = nn.MultiheadAttention(embed_dim=192, num_heads=8, dropout=0.1)  # output shape: (192, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)  # output shape: (384, 16, 16)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=384, momentum=0.9, eps=1e-4)  # output shape: (384, 16, 16)
        
        # SiLU Activation
        self.silu2 = nn.SiLU()  # output shape: (384, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (384, 8, 8)
        
        # Residual Connection 2
        self.residual2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1)  # output shape: (384, 8, 8)
        
        # Custom Self-Attention Mechanism 2
        self.self_attention2 = nn.MultiheadAttention(embed_dim=384, num_heads=8, dropout=0.1)  # output shape: (384, 8, 8)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=768, kernel_size=3, stride=1, padding=1)  # output shape: (768, 8, 8)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=768, momentum=0.9, eps=1e-4)  # output shape: (768, 8, 8)
        
        # SiLU Activation
        self.silu3 = nn.SiLU()  # output shape: (768, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (768, 4, 4)
        
        # Residual Connection 3
        self.residual3 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=1)  # output shape: (768, 4, 4)
        
        # Custom Self-Attention Mechanism 3
        self.self_attention3 = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1)  # output shape: (768, 4, 4)
        
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=768, out_channels=1536, kernel_size=3, stride=1, padding=1)  # output shape: (1536, 4, 4)
        
        # Batch Normalization 4
        self.bn4 = nn.BatchNorm2d(num_features=1536, momentum=0.9, eps=1e-4)  # output shape: (1536, 4, 4)
        
        # SiLU Activation
        self.silu4 = nn.SiLU()  # output shape: (1536, 4, 4)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (1536, 2, 2)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1)  # output shape: (1536*2*2)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=1536*2*2, out_features=768)  # output shape: (768)
        self.bn5 = nn.BatchNorm1d(num_features=768, momentum=0.9, eps=1e-4)  # output shape: (768)
        
        # SiLU Activation
        self.silu5 = nn.SiLU()  # output shape: (768)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=768, out_features=10)  # output shape: (10)

    def forward(self, x):
        # Apply self-attention mechanisms and max pooling
        x = self.pool1(self.silu1(self.bn1(self.conv1(x))))  # output shape: (192, 16, 16)
        x = self._apply_self_attention(self.self_attention1, x) + self.residual1(x)  # output shape: (192, 16, 16)
        
        x = self.pool2(self.silu2(self.bn2(self.conv2(x))))  # output shape: (384, 8, 8)
        x = self._apply_self_attention(self.self_attention2, x) + self.residual2(x)  # output shape: (384, 8, 8)
        
        x = self.pool3(self.silu3(self.bn3(self.conv3(x))))  # output shape: (768, 4, 4)
        x = self._apply_self_attention(self.self_attention3, x) + self.residual3(x)  # output shape: (768, 4, 4)
        
        x = self.pool4(self.silu4(self.bn4(self.conv4(x))))  # output shape: (1536, 2, 2)
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x)  # output shape: (1536*2*2)
        x = self.silu5(self.bn5(self.fc1(x)))  # output shape: (768)
        x = self.fc2(x)  # output shape: (10)
        
        return x
    
    def _apply_self_attention(self, attention_layer, x):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)  # Reshape to 4D tensor for attention layer
        attn_output, _ = attention_layer(query=x, key=x, value=x)
        attn_output = attn_output.permute(0, 3, 1, 2)  # Reshape back to 4D tensor
        return attn_output