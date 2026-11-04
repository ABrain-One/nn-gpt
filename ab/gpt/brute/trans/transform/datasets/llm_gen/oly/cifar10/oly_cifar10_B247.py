import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, mode='symmetric', fill=(199, 77, 171)), transforms.RandomRotation(degrees=17), transforms.ColorJitter(brightness=0.86, contrast=1.02, saturation=0.84, hue=0.03), transforms.RandomAutocontrast(p=0.9), transforms.RandomHorizontalFlip(), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(**norm)])