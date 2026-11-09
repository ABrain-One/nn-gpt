import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.76, 0.85), ratio=(1.05, 1.43)), transforms.RandomSolarize(threshold=148, p=0.15), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])