import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(50, 111, 220), padding_mode='symmetric'), transforms.RandomResizedCrop(64, scale=(0.8, 1.2), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.RandomPerspective(distortion_scale=0.05, p=0.5), transforms.RandomAutocontrast(p=0.25), transforms.RandomInvert(p=0.1), transforms.ToTensor(), transforms.Normalize(mean=[0.49139975, 0.48215827, 0.44653103], std=[0.24703233, 0.24348202, 0.26158708])])