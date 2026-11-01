import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.8), transforms.RandomApply([transforms.Grayscale(p=1.0)], p=0.2), transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.5), transforms.Pad(padding=3, fill=(125, 122, 113), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])