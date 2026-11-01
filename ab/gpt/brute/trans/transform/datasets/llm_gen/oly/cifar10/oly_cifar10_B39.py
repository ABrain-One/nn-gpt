import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(255, 190, 17), padding_mode='edge'), transforms.RandomResizedCrop(size=48, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(*norm)])