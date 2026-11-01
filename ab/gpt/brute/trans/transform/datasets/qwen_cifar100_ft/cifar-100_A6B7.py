import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=28), transforms.RandomPerspective(distortion_scale=0.1, p=0.33), transforms.ColorJitter(brightness=0.88, contrast=0.81, saturation=1.11, hue=0.02), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])