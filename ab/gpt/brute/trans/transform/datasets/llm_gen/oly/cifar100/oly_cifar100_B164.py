import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.31), transforms.RandomVerticalFlip(p=0.87), transforms.Pad(padding=0, fill=(91, 232, 21), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3), transforms.ToTensor(), transforms.Normalize(**norm)])