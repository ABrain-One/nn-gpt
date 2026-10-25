import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(0.75, 1.333)), transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.25, p=1.0)], p=0.5), transforms.RandomApply([transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-5.0, 5.0))], p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize(64), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])