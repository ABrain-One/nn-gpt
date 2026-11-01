import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5)] , p=0.2),
        transforms.RandomApply([transforms.Autocontrast()] , p=0.5),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])