import torch
import torchvision
import torchvision.transforms as transform
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Download and load training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Download and load test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Check if datasets are loaded correctly
if len(trainset) > 0 and len(testset) > 0:
    print("Loading successful!")
else:
    print("Loading failed.")