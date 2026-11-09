import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.75), transforms.RandomResizedCrop(32, scale=(0.6, 1.0), ratio=(0.8, 1.3)), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.2), transforms.RandomPerspective(distortion_scale=0.2, p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4401], std=[0.2673, 0.2564, 0.2762])])