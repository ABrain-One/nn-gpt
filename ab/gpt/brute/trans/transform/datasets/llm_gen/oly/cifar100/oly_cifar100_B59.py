import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(72), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.05), transforms.RandomAdjustSharpness(1.0, p=0.8), transforms.RandomApply([transforms.Posterize(4)], p=0.3), transforms.RandomApply([transforms.Equalize()], p=0.3), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(**norm)])