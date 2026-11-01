import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomRotation(10), transforms.Pad(padding=4, fill=(125, 123, 114), padding_mode='constant'), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])