## Role

You are an excellent vision model architect. You have much experience in designing vision models. You are good at improving the vision model code. You have a lot of experience of debugging the vision model code.

## Task

You are given:
1. The BEST performing code so far (reference implementation)
2. The CURRENT iteration's code and its evaluation result
3. History of previous improvement attempts and their results (learn from past experience!)

Analyze the problems and provide specific improvement suggestions for the next iteration.

- The evaluation result is a float number between 0 and 1, meaning the accuracy of the vision model.
- Sometimes the evaluation maybe a message string, meaning the error or other questions of the vision model. In this case, you should provide suggestions to fix the error.
- Your suggestions should help improve upon the BEST code, not just fix the current iteration's issues.
- IMPORTANT: Learn from the improvement history! Avoid repeating suggestions that didn't work, and build upon ideas that showed improvement.

## Improvement History (Recent Iterations)
### Iteration 83
- **Problem identified**: The accuracy is low or the error occurred due to the incorrect implementation of adaptive feature map aggregation and the integration of the feature pyramid network (FPN). Specifically, the current implementation concatenates feature maps into a 4D tensor, which is incompatible with subsequent operations expecting 2D or 3D inputs. Additionally, the spatial attention mechanism is not properly aligned with the aggregated feature maps, leading to a misalignment in dimensions.
- **Suggestion given**: [{'suggestion': 'Refactor Adaptive Feature Map Aggregation with Proper Dimensionality Reduction', 'details': 'Refactor the adaptive feature map aggregation to ensure that the feature maps are correctly aggregated into a 2D tensor before being passed to the subsequent layers. Instead of concatenating...
- **Result after applying**: error: Code validation failed: Validation error: ValueError: expected 2D or 3D input (got 4D input)

### Iteration 84
- **Problem identified**: The accuracy is low or the error occurred due to the incorrect implementation of the adaptive feature map aggregation and the spatial attention mechanism. The current implementation introduces a 4D tensor during the adaptive feature map aggregation step, which is incompatible with subsequent operations expecting 2D or 3D inputs. Additionally, the spatial attention mechanism is not properly integrated with the aggregated feature maps, leading to a misalignment in dimensions.
- **Suggestion given**: [{'suggestion': 'Refactor Adaptive Feature Map Aggregation with Proper Dimensionality Reduction', 'details': 'Refactor the adaptive feature map aggregation to ensure that the feature maps are correctly aggregated into a 2D tensor before being passed to the subsequent layers. Instead of using `torch....
- **Result after applying**: error: Code validation failed: Validation error: ValueError: expected 2D or 3D input (got 4D input)

### Iteration 85
- **Problem identified**: The accuracy is low or the error occurred due to the incorrect implementation of adaptive feature map aggregation and the spatial attention mechanism. Specifically, the current implementation introduces a 4D tensor during these steps, which is incompatible with subsequent operations expecting 2D or 3D inputs. Additionally, the spatial attention mechanism is not properly aligned with the aggregated feature maps, leading to a misalignment in dimensions.
- **Suggestion given**: [{'suggestion': 'Refactor Adaptive Feature Map Aggregation with Proper Dimensionality Reduction', 'details': 'Refactor the adaptive feature map aggregation to ensure that the feature maps are correctly aggregated into a 2D tensor before being passed to the subsequent layers. Instead of using `torch....
- **Result after applying**: error: Code validation failed: Validation error: ValueError: expected 2D or 3D input (got 4D input)

### Iteration 88
- **Problem identified**: The accuracy is low or the error occurred primarily due to the incorrect handling of 4D tensors in the adaptive feature map aggregation and the spatial attention mechanism. The current implementation introduces 4D tensors during these steps, which is incompatible with subsequent operations expecting 2D or 3D inputs. Additionally, there is a misalignment in dimensions between the spatial attention mechanism and the aggregated feature maps.
- **Suggestion given**: [{'suggestion': 'Refactor Adaptive Feature Map Aggregation with Proper Dimensionality Reduction', 'details': 'Refactor the adaptive feature map aggregation to ensure that the feature maps are correctly aggregated into a 2D tensor before being passed to the subsequent layers. Specifically, use `nn.Ad...
- **Result after applying**: error: Code validation failed: Validation error: ValueError: expected 2D or 3D input (got 4D input)

### Iteration 89
- **Problem identified**: The accuracy is low or the error occurred primarily due to the incorrect handling of 4D tensors in the adaptive feature map aggregation and the spatial attention mechanism. The current implementation introduces 4D tensors during these steps, which is incompatible with subsequent operations expecting 2D or 3D inputs. Additionally, there is a misalignment in dimensions between the spatial attention mechanism and the aggregated feature maps.
- **Suggestion given**: [{'suggestion': 'Refactor Adaptive Feature Map Aggregation with Proper Dimensionality Reduction', 'details': 'Refactor the adaptive feature map aggregation to ensure that the feature maps are correctly aggregated into 2D or 3D tensors before being passed to the subsequent layers. Specifically, inste...
- **Result after applying**: error: Code validation failed: Validation error: ValueError: expected 2D or 3D input (got 4D input)


## Best Code (Reference - Accuracy: 66.41%)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) # output shape: (32, 32, 32)
        
        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(num_features=32, momentum=0.9, eps=1e-5) # output shape: (32, 32, 32)
        
        # SiLU Activation
        self.silu1 = nn.SiLU() # output shape: (32, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (32, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # output shape: (64, 16, 16)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5) # output shape: (64, 16, 16)
        
        # SiLU Activation
        self.silu2 = nn.SiLU() # output shape: (64, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (64, 8, 8)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # output shape: (128, 8, 8)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=128, momentum=0.9, eps=1e-5) # output shape: (128, 8, 8)
        
        # SiLU Activation
        self.silu3 = nn.SiLU() # output shape: (128, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (128, 4, 4)
        
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # output shape: (256, 4, 4)
        
        # Batch Normalization 4
        self.bn4 = nn.BatchNorm2d(num_features=256, momentum=0.9, eps=1e-5) # output shape: (256, 4, 4)
        
        # SiLU Activation
        self.silu4 = nn.SiLU() # output shape: (256, 4, 4)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (256, 2, 2)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=256*2*2, out_features=1024) # output shape: (1024)
        
        # Batch Normalization 5
        self.bn5 = nn.BatchNorm1d(num_features=1024, momentum=0.9, eps=1e-5) # output shape: (1024)
        
        # SiLU Activation
        self.silu5 = nn.SiLU() # output shape: (1024)
        
        # Conditional Stochastic Depth
        self.drop_path = nn.ModuleList([nn.Identity() for _ in range(4)]) # output shape: (1024)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=1024, out_features=10) # output shape: (10)
        
        # Adaptive Attention Mechanism
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1), 
            nn.BatchNorm2d(num_features=32),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 16, 16)
        
        self.attention2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), 
            nn.BatchNorm2d(num_features=64),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 8, 8)
        
        self.attention3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1), 
            nn.BatchNorm2d(num_features=128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 4, 4)
        
        self.attention4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1), 
            nn.BatchNorm2d(num_features=256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 2, 2)
    
    def forward(self, x):
        # Apply attention mechanisms
        x = self.pool1(self.silu1(self.bn1(self.conv1(x)))) # output shape: (32, 16, 16)
        att1 = self.attention1(x)
        x = att1 * x + x # output shape: (32, 16, 16)
        
        x = self.pool2(self.silu2(self.bn2(self.conv2(x)))) # output shape: (64, 8, 8)
        att2 = self.attention2(x)
        x = att2 * x + x # output shape: (64, 8, 8)
        
        x = self.pool3(self.silu3(self.bn3(self.conv3(x)))) # output shape: (128, 4, 4)
        att3 = self.attention3(x)
        x = att3 * x + x # output shape: (128, 4, 4)
        
        x = self.pool4(self.silu4(self.bn4(self.conv4(x)))) # output shape: (256, 2, 2)
        att4 = self.attention4(x)
        x = att4 * x + x # output shape: (256, 2, 2)
        
        x = x.view(-1, 256*2*2) # output shape: (1024)
        x = self.silu5(self.bn5(self.fc1(x))) # output shape: (1024)
        
        # Apply conditional stochastic depth
        for i, drop_path in enumerate(self.drop_path):
            if i < 3:  # Apply stochastic depth only to the first three fully connected layers
                x = drop_path(x)
        
        x = self.fc2(x) # output shape: (10)
        
        return x

## Current Iteration Code (Accuracy: Failed)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) # output shape: (64, 32, 32)
        
        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5) # output shape: (64, 32, 32)
        
        # SiLU Activation
        self.silu1 = nn.SiLU() # output shape: (64, 32, 32)
        
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (64, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # output shape: (128, 16, 16)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=128, momentum=0.9, eps=1e-5) # output shape: (128, 16, 16)
        
        # SiLU Activation
        self.silu2 = nn.SiLU() # output shape: (128, 16, 16)
        
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (128, 8, 8)
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # output shape: (256, 8, 8)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=256, momentum=0.9, eps=1e-5) # output shape: (256, 8, 8)
        
        # SiLU Activation
        self.silu3 = nn.SiLU() # output shape: (256, 8, 8)
        
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (256, 4, 4)
        
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) # output shape: (512, 4, 4)
        
        # Batch Normalization 4
        self.bn4 = nn.BatchNorm2d(num_features=512, momentum=0.9, eps=1e-5) # output shape: (512, 4, 4)
        
        # SiLU Activation
        self.silu4 = nn.SiLU() # output shape: (512, 4, 4)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (512, 2, 2)
        
        # Feature Pyramid Network (FPN)
        self.lateral_conv1 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=1) # output shape: (512, 16, 16)
        self.lateral_conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1) # output shape: (512, 8, 8)
        self.lateral_conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1) # output shape: (512, 4, 4)
        
        self.fpn_up1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2) # output shape: (512, 32, 32)
        self.fpn_up2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=4) # output shape: (512, 64, 64)
        
        # Adaptive Feature Map Aggregation
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # output shape: (512, 1, 1)
        self.flatten = nn.Flatten() # output shape: (512)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=512, out_features=512) # output shape: (512)
        self.bn1 = nn.BatchNorm1d(num_features=512, momentum=0.9, eps=1e-5) # output shape: (512)
        self.silu1 = nn.SiLU() # output shape: (512)
        self.dropout1 = nn.Dropout(p=0.4) # output shape: (512)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=512, out_features=10) # output shape: (10)
        
        # Spatial Attention Mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1), 
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (512, 2, 2)
    
    def forward(self, x):
        # Forward pass through convolutional layers
        x1 = self.pool1(self.silu1(self.bn1(self.conv1(x)))) # output shape: (64, 16, 16)
        x2 = self.pool2(self.silu2(self.bn2(self.conv2(x1)))) # output shape: (128, 8, 8)
        x3 = self.pool3(self.silu3(self.bn3(self.conv3(x2)))) # output shape: (256, 4, 4)
        x4 = self.pool4(self.silu4(self.bn4(self.conv4(x3)))) # output shape: (512, 2, 2)
        
        # Feature Pyramid Network (FPN)
        p1 = self.lateral_conv1(x1) # output shape: (512, 16, 16)
        p2 = self.lateral_conv2(x2) # output shape: (512, 8, 8)
        p3 = self.lateral_conv3(x3) # output shape: (512, 4, 4)
        
        p2 = p2 + self.fpn_up1(p1) # output shape: (512, 32, 32)
        p3 = p3 + self.fpn_up2(p2) # output shape: (512, 64, 64)
        
        # Adaptive Feature Map Aggregation
        p1 = self.adaptive_avg_pool(p1).view(-1, 512) # output shape: (512)
        p2 = self.adaptive_avg_pool(p2).view(-1, 512) # output shape: (512)
        p3 = self.adaptive_avg_pool(p3).view(-1, 512) # output shape: (512)
        p4 = self.adaptive_avg_pool(x4).view(-1, 512) # output shape: (512)
        
        x = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1), p3.unsqueeze(1), p4.unsqueeze(1)], dim=1) # output shape: (batch_size, 4, 512)
        x = x.mean(dim=1) # output shape: (batch_size, 512)
        
        # Apply spatial attention mechanism
        x = x * self.spatial_attention(x1.unsqueeze(1).unsqueeze(1)).squeeze(3).squeeze(2) + x
        
        # Fully Connected Layers
        x = self.silu1(self.bn1(self.fc1(x))) # output shape: (512)
        x = self.dropout1(x)
        x = self.fc2(x) # output shape: (10)
        
        return x

## Feedback from Evaluator
Code validation failed: Validation error: ValueError: expected 2D or 3D input (got 4D input)

## Tips

- **Learn from history**: Review what was tried before and what results it produced. Don't repeat failed approaches.
- Compare the current code with the best code to understand what changes led to performance differences.
- Analyze the possible reasons why the accuracy is low or why the error occurred.
- Provide specific, actionable improvement suggestions that build upon the BEST code.
- You have a lot of knowledge, so the inspiration can come from any subject, such as computer science, philosophy, economics, biology, etc.
- Focus on concrete architectural changes, hyperparameter suggestions, or bug fixes.

## Output format
```json
{
    "reason": "The reason why the accuracy is low or the error occurred",
    "inspiration": "The inspiration from any subject, such as computer science, philosophy, economics, biology, etc.",
    "improvement_suggestions": "Specific, actionable suggestions on how to improve upon the best code. Be detailed and concrete."
}
```

