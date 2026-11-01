import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.63, 0.92), ratio=(1.09, 1.88)), transforms.RandomSolarize(threshold=251, p=0.21), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])