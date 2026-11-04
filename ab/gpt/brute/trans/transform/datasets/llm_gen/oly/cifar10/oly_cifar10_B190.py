import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(40), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomPerspective(distortion_scale=0.2, p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])