import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(74, scale=(0.6, 1.0), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1), transforms.RandomGrayscale(p=0.25), transforms.RandomSolarize(threshold=240, p=0.7), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4401], std=[0.2675, 0.2565, 0.2762])])