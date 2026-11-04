import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.76, 0.87), ratio=(1.02, 1.53)), transforms.RandomAdjustSharpness(sharpness_factor=1.03, p=0.15), transforms.RandomPerspective(distortion_scale=0.11, p=0.43), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])