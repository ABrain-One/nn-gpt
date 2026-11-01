import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.875, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(hue=0.1, saturation=0.2, brightness=0.2, contrast=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))], p=0.5),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994])
    ])