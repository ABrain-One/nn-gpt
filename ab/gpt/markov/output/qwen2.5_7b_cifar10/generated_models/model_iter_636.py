class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Block 1
        self.block1 = ResidualBlock(in_channels=3, out_channels=192, kernel_size=3, stride=1, padding=1)  # output shape: (192, 32, 32)
        
        # Convolutional Block 2
        self.block2 = ResidualBlock(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)  # output shape: (384, 16, 16)
        
        # Convolutional Block 3
        self.block3 = ResidualBlock(in_channels=384, out_channels=768, kernel_size=3, stride=1, padding=1)  # output shape: (768, 8, 8)
        
        # Convolutional Block 4
        self.block4 = ResidualBlock(in_channels=768, out_channels=1536, kernel_size=3, stride=1, padding=1)  # output shape: (1536, 4, 4)
        
        # Convolutional Block 5
        self.block5 = ResidualBlock(in_channels=1536, out_channels=1536, kernel_size=3, stride=1, padding=1)  # output shape: (1536, 4, 4)
        
        # Convolutional Block 6
        self.block6 = ResidualBlock(in_channels=1536, out_channels=1536, kernel_size=3, stride=1, padding=1)  # output shape: (1536, 4, 4)
        
        # Convolutional Block 7
        self.block7 = ResidualBlock(in_channels=1536, out_channels=1536, kernel_size=3, stride=1, padding=1)  # output shape: (1536, 4, 4)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1)  # output shape: (1536*4*4)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=1536*4*4, out_features=768)  # output shape: (768)
        self.bn5 = nn.BatchNorm1d(num_features=768, momentum=0.9, eps=1e-4)  # output shape: (768)
        
        # SiLU Activation
        self.silu5 = nn.SiLU()  # output shape: (768)
        
        # Dropout Regularization
        self.dropout1 = nn.Dropout(p=0.3)  # output shape: (768)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=768, out_features=10)  # output shape: (10)

    def forward(self, x):
        x = self.block1(x)  # output shape: (192, 32, 32)
        x = self.block2(x)  # output shape: (384, 16, 16)
        x = self.block3(x)  # output shape: (768, 8, 8)
        x = self.block4(x)  # output shape: (1536, 4, 4)
        x = self.block5(x)  # output shape: (1536, 4, 4)
        x = self.block6(x)  # output shape: (1536, 4, 4)
        x = self.block7(x)  # output shape: (1536, 4, 4)
        
        x = self.flatten(x)  # output shape: (1536*4*4)
        x = self.silu5(self.bn5(self.fc1(x)))  # output shape: (768)
        x = self.dropout1(x)  # output shape: (768)
        x = self.fc2(x)  # output shape: (10)
        
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        
        # Convolutional Layer
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)  # output shape: (out_channels, H, W)
        
        # Batch Normalization
        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9, eps=1e-4)  # output shape: (out_channels, H, W)
        
        # SiLU Activation
        self.silu = nn.SiLU()  # output shape: (out_channels, H, W)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # output shape: (out_channels, H/2, W/2)
        
        # Self-Attention Mechanism
        self.self_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=8, dropout=0.3)  # output shape: (out_channels, H/2, W/2)
        
        # Residual Connection
        self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)  # output shape: (out_channels, H, W)
    
    def forward(self, x):
        identity = x.clone()  # Save the input for the residual connection
        
        # Apply convolution, batch normalization, activation, and self-attention
        x = self.silu(self.bn(self.conv(x)))  # output shape: (out_channels, H/2, W/2)
        x = self.apply_self_attention(x) + self.residual(identity)  # output shape: (out_channels, H/2, W/2)
        x = self.pool(x)  # output shape: (out_channels, H/4, W/4)
        
        return x
    
    def apply_self_attention(self, x):
        b, c, h, w = x.size()
        if x.dim() != 4:
            raise ValueError(f"Input tensor should be 4D, but got {x.dim()}D tensor")
        
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h*w, c)  # Reshape to 3D tensor for attention layer
        x = self.bn(x)  # Apply batch normalization before self-attention
        attn_output, _ = self.self_attention(query=x, key=x, value=x)
        attn_output = attn_output.view(b, c, h, w).permute(0, 3, 1, 2).contiguous()  # Reshape back to the original 4D format
        return attn_output