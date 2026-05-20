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
### Iteration 1
- **Problem identified**: The accuracy is low because the architecture of the current model is similar to the best code, but there might be subtle differences in the implementation or the learning process that could be causing the performance to stagnate. Additionally, the model might benefit from more advanced techniques or architectural changes to improve its ability to learn complex features.
- **Suggestion given**: [{'suggestion': 'Increase the depth of the network by adding more convolutional layers with increasing filter sizes. This can help the model capture more complex features.', 'details': 'Add two more convolutional layers after the initial two, each with increasing filter sizes (e.g., 128 filters in t...
- **Result after applying**: error: Code validation failed: Validation error: RuntimeError: Given groups=1, weight of size [256, 64, 1, 1], expected input[2, 3, 32, 32] to have 64 channels, but got 3 channels instead

### Iteration 2
- **Problem identified**: The error occurred due to a mismatch in the channel dimensions when attempting to apply a residual connection. Additionally, the architecture of the current model introduces unnecessary complexity without clear benefits, leading to potential overfitting or suboptimal feature extraction.
- **Suggestion given**: [{'suggestion': 'Remove the residual connection and simplify the architecture to match the best-performing code more closely.', 'details': 'The current model introduces a residual connection that adds complexity and may lead to issues like the one encountered during validation. Removing it will make...
- **Result after applying**: accuracy: 47.50%

### Iteration 3
- **Problem identified**: The accuracy is low because the current architecture includes additional convolutional layers and fully connected layers compared to the best code. These extra layers introduce more parameters and complexity, which can lead to overfitting, especially if the dataset is small or the training process is not well-tuned. Additionally, the current implementation has a mismatch in the fully connected layer dimensions, which results in an incorrect number of input features for the final layer.
- **Suggestion given**: [{'suggestion': 'Reduce the number of convolutional and fully connected layers to match the best code more closely.', 'details': 'Remove the third convolutional layer (conv3) and the corresponding batch normalization and pooling layers. Also, adjust the second fully connected layer (fc2) to have 512...
- **Result after applying**: accuracy: 48.40%

### Iteration 5
- **Problem identified**: The accuracy is low primarily due to the mismatch in the fully connected layer dimensions. Specifically, the current implementation views the output of the max pooling layer incorrectly, resulting in an incorrect number of input features for the fully connected layer. This misalignment can cause the model to fail to learn effectively, leading to suboptimal performance. Additionally, there is no explicit dropout layer after the second fully connected layer, which can help prevent overfitting.
- **Suggestion given**: [{'suggestion': 'Correct the view operation in the forward pass to match the best code.', 'details': 'Change `x = x.view(-1, 64*8*8)` to `x = x.view(-1, 64*8*8)`. This ensures that the input to the first fully connected layer matches the expected dimension of 64 * 8 * 8.'}, {'suggestion': 'Add a dro...
- **Result after applying**: accuracy: 10.00%


## Best Code (Reference - Accuracy: 48.40%)
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=512) # output shape: (512)
        
        # Dropout 1
        self.dropout1 = nn.Dropout(p=0.5) # output shape: (512)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=512, out_features=10) # output shape: (10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x)))) # output shape: (32, 16, 16)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x)))) # output shape: (64, 8, 8)
        x = x.view(-1, 64*8*8) # output shape: (16384)
        x = self.dropout1(self.relu1(self.fc1(x))) # output shape: (512)
        x = self.fc2(x) # output shape: (10)
        return x

## Current Iteration Code (Accuracy: 10.00%)
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
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=512) # output shape: (512)
        
        # Dropout 1
        self.dropout1 = nn.Dropout(p=0.5) # output shape: (512)
        
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=512, out_features=10) # output shape: (10)
        
        # Dropout 2
        self.dropout2 = nn.Dropout(p=0.5) # output shape: (10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x)))) # output shape: (32, 16, 16)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x)))) # output shape: (64, 8, 8)
        
        x = x.view(-1, 64*8*8) # output shape: (2048)
        x = self.dropout1(self.relu1(self.fc1(x))) # output shape: (512)
        x = self.dropout2(self.relu1(self.fc2(x))) # output shape: (10)
        return x

## Feedback from Evaluator
Training successful. Test accuracy: 10.00%

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

