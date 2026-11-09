import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=(-5, 5)), transforms.CenterCrop(27), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))], p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])