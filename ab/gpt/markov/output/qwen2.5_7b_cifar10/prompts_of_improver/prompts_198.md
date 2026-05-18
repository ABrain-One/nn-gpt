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
### Iteration 193
- **Problem identified**: The accuracy is low primarily because of the architectural differences and the use of depthwise separable convolutions without proper handling of group parameters. Additionally, the use of spatial pyramid pooling (SPP) and adaptive average pooling (AAP) introduces complexity that might disrupt the feature flow and model stability. The error indicates that the out_channels must be divisible by the groups parameter, which is violated in the current implementation.
- **Suggestion given**: [{'suggestion': 'Correct Depthwise Separable Convolution Implementation', 'details': 'Ensure that the depthwise separable convolutions are correctly implemented with proper group parameters. Specifically, the out_channels in the second convolution should be equal to the input channels to avoid viola...
- **Result after applying**: error: Code validation failed: Validation error: ValueError: out_channels must be divisible by groups

### Iteration 194
- **Problem identified**: The accuracy is low and the error occurred primarily due to incorrect implementation of depthwise separable convolutions. The `groups` parameter is not set correctly, leading to the `out_channels` being non-divisible by `groups`. Additionally, the current architecture lacks some of the key components found in the best code, such as attention mechanisms and consistent padding settings. This has disrupted the feature flow and model stability, leading to lower performance.
- **Suggestion given**: [{'suggestion': 'Correct Depthwise Separable Convolution Implementation', 'details': 'Ensure that the depthwise separable convolutions are correctly implemented with proper group parameters. Specifically, the `groups` parameter should match the input channels for the first convolution in each depthw...
- **Result after applying**: error: Code validation failed: Validation error: ValueError: out_channels must be divisible by groups

### Iteration 195
- **Problem identified**: The accuracy is low and the error occurred primarily due to incorrect implementation of depthwise separable convolutions. The `groups` parameter is not set correctly, leading to the `out_channels` being non-divisible by `groups`. Additionally, the current architecture lacks some of the key components found in the best code, such as consistent padding settings and batch normalization, which disrupt the feature flow and model stability. The use of spatial pyramid pooling (SPP) and inconsistent attention mechanisms also contribute to lower performance.
- **Suggestion given**: [{'suggestion': 'Correct Depthwise Separable Convolution Implementation', 'details': 'Ensure that the depthwise separable convolutions are correctly implemented with proper group parameters. Specifically, the `groups` parameter should match the input channels for the first convolution in each depthw...
- **Result after applying**: error: Code validation failed: Validation error: ValueError: out_channels must be divisible by groups

### Iteration 196
- **Problem identified**: The accuracy is low primarily because of the incorrect implementation of depthwise separable convolutions and the misuse of adaptive average pooling (AAP). The `out_channels` in the second convolution of each depthwise separable convolution layer must be divisible by the `groups` parameter, which is not satisfied in the current implementation. Additionally, the use of AAP followed by concatenation and flattening leads to an unexpected increase in the number of features, disrupting the feature flow and model stability.
- **Suggestion given**: [{'suggestion': 'Correct Depthwise Separable Convolution Implementation', 'details': 'Ensure that the depthwise separable convolutions are correctly implemented with proper group parameters. Specifically, the `out_channels` in the second convolution should match the input channels to ensure divisibi...
- **Result after applying**: error: Code validation failed: Validation error: ValueError: out_channels must be divisible by groups

### Iteration 197
- **Problem identified**: The accuracy is low primarily because of the incorrect implementation of depthwise separable convolutions. Specifically, the `groups` parameter is not set correctly, leading to the `out_channels` being non-divisible by `groups`. Additionally, the current architecture lacks consistent padding settings and batch normalization, which disrupt the feature flow and model stability. The use of depthwise separable convolutions without proper handling of group parameters and the misuse of adaptive average pooling (AAP) also contribute to lower performance.
- **Suggestion given**: [{'suggestion': 'Correct Depthwise Separable Convolution Implementation', 'details': 'Ensure that the depthwise separable convolutions are correctly implemented with proper group parameters. Specifically, the `groups` parameter should match the input channels for the first convolution in each depthw...
- **Result after applying**: error: Code validation failed: Validation error: ValueError: out_channels must be divisible by groups


## Best Code (Reference - Accuracy: 69.71%)
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
        
        # Attention Mechanism 1
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), 
            nn.BatchNorm2d(num_features=64, momentum=0.9, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        ) # output shape: (1, 16, 16)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # output shape: (128, 16, 16)
        
        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(num_features=128, momentum=0.9, eps=1e-5) # output shape: (128, 16, 16)
        
        # SiLU Activation
        self.silu2 = nn.SiLU() # output shape: (128, 16, 16)
        
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
        
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # output shape: (256, 8, 8)
        
        # Batch Normalization 3
        self.bn3 = nn.BatchNorm2d(num_features=256, momentum=0.9, eps=1e-5) # output shape: (256, 8, 8)
        
        # SiLU Activation
        self.silu3 = nn.SiLU() # output shape: (256, 8, 8)
        
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
        
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) # output shape: (512, 4, 4)
        
        # Batch Normalization 4
        self.bn4 = nn.BatchNorm2d(num_features=512, momentum=0.9, eps=1e-5) # output shape: (512, 4, 4)
        
        # SiLU Activation
        self.silu4 = nn.SiLU() # output shape: (512, 4, 4)
        
        # Max Pooling 4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output shape: (512, 2, 2)
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1) # output shape: (512*2*2)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=512*2*2, out_features=1024) # output shape: (1024)
        
        # Batch Normalization 5
        self.bn5 = nn.BatchNorm1d(num_features=1024, momentum=0.9, eps=1e-5) # output shape: (1024)
        
        # SiLU Activation
        self.silu5 = nn.SiLU() # output shape: (1024)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=1024, out_features=10) # output shape: (10)
    
    def forward(self, x):
        # Apply attention mechanisms and max pooling
        x = self.pool1(self.silu1(self.bn1(self.conv1(x)))) # output shape: (64, 16, 16)
        att1 = self.attention1(x)
        x = att1 * x + x # output shape: (64, 16, 16)
        
        x = self.pool2(self.silu2(self.bn2(self.conv2(x)))) # output shape: (128, 8, 8)
        att2 = self.attention2(x)
        x = att2 * x + x # output shape: (128, 8, 8)
        
        x = self.pool3(self.silu3(self.bn3(self.conv3(x)))) # output shape: (256, 4, 4)
        att3 = self.attention3(x)
        x = att3 * x + x # output shape: (256, 4, 4)
        
        x = self.pool4(self.silu4(self.bn4(self.conv4(x)))) # output shape: (512, 2, 2)
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x) # output shape: (512*2*2)
        x = self.silu5(self.bn5(self.fc1(x))) # output shape: (1024)
        x = self.fc2(x) # output shape: (10)
        
        return x

## Current Iteration Code (Accuracy: Failed)
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
        
        # Flatten and pass through fully connected layers
        self.flatten = nn.Flatten(start_dim=1) # output shape: (512*4*4)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=512*4*4, out_features=1024) # output shape: (1024)
        
        # Batch Normalization 5
        self.bn5 = nn.BatchNorm1d(num_features=1024, momentum=0.9, eps=1e-5) # output shape: (1024)
        
        # SiLU Activation
        self.silu5 = nn.SiLU() # output shape: (1024)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=1024, out_features=10) # output shape: (10)
    
    def forward(self, x):
        # Apply attention mechanisms and max pooling
        x = self.pool1(self.dsc1(x)) # output shape: (64, 16, 16)
        att1 = self.attention1(x)
        x = att1 * x + x # output shape: (64, 16, 16)
        
        x = self.pool2(self.dsc2(x)) # output shape: (128, 8, 8)
        att2 = self.attention2(x)
        x = att2 * x + x # output shape: (128, 8, 8)
        
        x = self.pool3(self.dsc3(x)) # output shape: (256, 4, 4)
        att3 = self.attention3(x)
        x = att3 * x + x # output shape: (256, 4, 4)
        
        x = self.dsc4(x) # output shape: (512, 4, 4)
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x) # output shape: (512*4*4)
        x = self.silu5(self.bn5(self.fc1(x))) # output shape: (1024)
        x = self.fc2(x) # output shape: (10)
        
        return x

## Feedback from Evaluator
Code validation failed: Validation error: ValueError: out_channels must be divisible by groups

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

