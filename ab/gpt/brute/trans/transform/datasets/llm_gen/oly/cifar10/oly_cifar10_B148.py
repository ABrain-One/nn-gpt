import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, mode='reflect', fill=(124, 124, 115)), transforms.RandomSolarize(threshold=120, p=0.8), transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201])])