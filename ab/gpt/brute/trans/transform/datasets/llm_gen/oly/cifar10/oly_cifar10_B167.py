import torch
from torchvision import transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.8), transforms.Pad(padding=2, fill=(104.545, 116.546, 129.188), padding_mode='constant'), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.5)], p=0.5), transforms.RandomCrop(32), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[104.545, 116.546, 129.188], std=[66.935, 63.022, 67.644])])