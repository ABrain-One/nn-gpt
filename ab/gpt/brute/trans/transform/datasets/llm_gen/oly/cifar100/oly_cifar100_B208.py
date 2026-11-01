import torch
import torchvision.transforms as transforms

def transform(norm):
    mean, std = norm
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(p=0.72), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2), transforms.RandomEqualize(p=0.6), transforms.RandomSolarize(threshold=14, p=0.3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean, std)])