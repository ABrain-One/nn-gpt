import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.2), ratio=(0.9, 1.1)), transforms.ColorJitter(brightness=0.9, contrast=1.1, saturation=0.9, hue=0.05), transforms.RandomVerticalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=(0.8, 1.4)), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.4386, 0.4498, 0.4705], std=[0.2863, 0.2792, 0.2713])])