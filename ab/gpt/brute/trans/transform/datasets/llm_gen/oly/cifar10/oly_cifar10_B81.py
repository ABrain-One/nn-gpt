import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.66), transforms.RandomAdjustSharpness(sharpness_factor=0.91, p=0.7), transforms.RandomAutocontrast(p=0.45), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*norm)])