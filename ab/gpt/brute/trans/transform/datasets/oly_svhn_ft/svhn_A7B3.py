import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.75), transforms.RandomPerspective(distortion_scale=0.15, p=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])