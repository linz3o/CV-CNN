import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_v1(nn.Module):
    def __init__(self):
        super(LeNet5_v1, self).__init__()
        
        # First Convolutional Layer (6 filters, 5x5 kernel)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  
        
        # Second Convolutional Layer (16 filters, 5x5 kernel)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
        
        # **New Third Convolutional Layer** (32 filters, 3x3 kernel)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)  

        # MaxPooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 3 * 3, 120)  # Adjusted input size due to extra conv layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Conv1 + ReLU
        x = self.pool(x)           # Pooling
        
        x = F.relu(self.conv2(x))  # Conv2 + ReLU
        x = self.pool(x)           # Pooling
        
        x = F.relu(self.conv3(x))  # **New Conv3 + ReLU**
        x = self.pool(x)           # Pooling

        x = torch.flatten(x, 1)  # Flatten the tensor for FC layers

        x = F.relu(self.fc1(x))  # Fully Connected 1 + ReLU
        x = F.relu(self.fc2(x))  # Fully Connected 2 + ReLU
        x = self.fc3(x)          # Output Layer (no activation since CrossEntropyLoss includes softmax)

        return x

# Example usage:
model = LeNet5_v1()
print(model)
