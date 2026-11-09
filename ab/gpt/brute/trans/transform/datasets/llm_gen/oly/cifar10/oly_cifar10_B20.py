import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(3, fill=(53, 7, 5), padding_mode='constant'), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(p=0.6), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(degrees=5), transforms.GaussianBlur(kernel_size=3, sigma=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])