import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=8, padding_mode='reflect'), transforms.RandomInvert(p=0.6), transforms.RandomAdjustSharpness(1.85, p=0.4), transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])