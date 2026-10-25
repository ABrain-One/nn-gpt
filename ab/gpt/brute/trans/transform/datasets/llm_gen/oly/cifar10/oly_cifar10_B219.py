import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(60, 137, 249), padding_mode='edge'), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.2), aspect_ratio=(0.8, 2.0)), transforms.RandomRotation(degrees=10), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])