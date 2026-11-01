import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.73, 0.97), ratio=(1.1, 2.13)), transforms.RandomGrayscale(p=0.28), transforms.RandomAdjustSharpness(sharpness_factor=1.17, p=0.47), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])