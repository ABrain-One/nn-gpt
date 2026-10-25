import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(0.9, 1.16, 1.01, 0.04), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])