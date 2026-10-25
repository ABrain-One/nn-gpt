import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=29), transforms.RandomPerspective(distortion_scale=0.18, p=0.37), transforms.RandomHorizontalFlip(p=0.33), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])