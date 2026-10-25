import torch
import torchvision.transforms as transforms

def transform(norm):
    mean, std = norm
    return transforms.Compose([transforms.Pad(padding=4, fill=tuple(mean), padding_mode='constant'), transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=1.0)], p=0.5), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])