import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.75, 1.0), ratio=(0.8, 1.2)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.RandomPosterize(4)], p=0.5), transforms.RandomApply([transforms.RandomSolarize(threshold=63)], p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])