import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.75, 1.333)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomSolarize(threshold=180, p=0.3), transforms.RandomEqualize(p=0.3), transforms.ToTensor(), transforms.Normalize(*norm)])