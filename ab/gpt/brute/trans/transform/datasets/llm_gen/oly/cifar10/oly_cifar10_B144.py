import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.3), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8), transforms.GaussianBlur(kernel_size=3, sigma=1.5, p=0.2), transforms.RandomPerspective(distortion_scale=0.4, p=0.5), transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])