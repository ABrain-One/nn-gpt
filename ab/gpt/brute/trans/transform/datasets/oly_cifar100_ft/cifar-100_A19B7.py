import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=0.85, p=0.16), transforms.RandomAutocontrast(p=0.12), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.481, 0.461, 0.7), (0.157, 0.228, 0.721))])