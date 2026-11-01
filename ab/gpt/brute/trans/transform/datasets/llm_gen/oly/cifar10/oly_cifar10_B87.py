import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.75, 1.5), interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), transforms.RandomGrayscale(p=0.5), transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(*norm)])