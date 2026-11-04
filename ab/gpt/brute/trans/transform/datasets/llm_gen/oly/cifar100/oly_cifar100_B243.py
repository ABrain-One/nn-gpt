import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(40), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(norm[0], norm[1], norm[2])])