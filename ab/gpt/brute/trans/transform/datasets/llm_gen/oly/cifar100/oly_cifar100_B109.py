import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.9, 1.1)), transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=0.5), transforms.RandomEqualize(p=0.5), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.486, 0.44], std=[0.267, 0.256, 0.276])])