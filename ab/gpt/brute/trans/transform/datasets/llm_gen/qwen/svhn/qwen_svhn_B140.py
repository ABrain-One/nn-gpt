import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(135, 135, 135), padding_mode='symmetric'), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(-10, 10)), transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.5, 1.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])