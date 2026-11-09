import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=32), transforms.RandomSolarize(threshold=51, p=0.46), transforms.RandomResizedCrop(size=32, scale=(0.58, 0.97), ratio=(1.0, 1.79)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])