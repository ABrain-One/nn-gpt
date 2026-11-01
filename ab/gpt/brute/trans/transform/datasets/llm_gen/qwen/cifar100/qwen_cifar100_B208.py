import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.8), transforms.RandomRotation(degrees=10), transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 1.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])