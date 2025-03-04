import torch
import torchvision
import torchvision.transforms as transform
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Download and load training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split the training set into training and validation sets
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])

# Download and load test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Check if datasets are loaded correctly
if len(trainset) > 0 and len(testset) > 0:
    print("Loading successful!")
else:
    print("Loading failed.")