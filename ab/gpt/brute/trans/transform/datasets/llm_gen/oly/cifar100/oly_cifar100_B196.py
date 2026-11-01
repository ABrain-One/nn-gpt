import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.65, 0.93), ratio=(0.97, 1.41)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=5, sigma=(0.88, 1.37)), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*norm)])