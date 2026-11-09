import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=20, fill=(135, 53, 187), padding_mode='constant'), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(degrees=(-15, 15)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4388, 0.4475, 0.4758], std=[0.2404, 0.2334, 0.2504])])