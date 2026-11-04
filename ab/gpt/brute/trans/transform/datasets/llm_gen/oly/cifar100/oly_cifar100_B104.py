import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.0)], p=0.5), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(**norm)])