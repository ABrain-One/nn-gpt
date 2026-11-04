import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.76, 0.87), ratio=(0.9, 1.42)), transforms.RandomPerspective(distortion_scale=0.25, p=0.33), transforms.RandomAdjustSharpness(sharpness_factor=0.87, p=0.62), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])