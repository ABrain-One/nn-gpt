import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomApply([transforms.RandomHorizontalFlip(p=1.0)], p=0.5), transforms.RandomApply([transforms.RandomVerticalFlip(p=1.0)], p=0.5), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])