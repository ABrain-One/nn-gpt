import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.7), transforms.RandomInvert(p=0.82), transforms.RandomPerspective(distortion_scale=0.13, p=0.69), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])