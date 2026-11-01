import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.5, 2.0), ratio=(0.9, 1.1)), transforms.RandomRotation(15), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5), transforms.RandomApply([transforms.Posterize(4)], p=0.5), transforms.ToTensor(), transforms.Normalize(mean=norm[0], std=norm[1])])