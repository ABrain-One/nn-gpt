import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(degrees=15), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.5)], p=0.5), transforms.RandomPosterize(bits=5, p=0.5), transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.8), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])