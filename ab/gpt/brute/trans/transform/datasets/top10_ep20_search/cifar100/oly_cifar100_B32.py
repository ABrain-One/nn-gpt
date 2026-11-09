import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.84, 1.0), ratio=(0.75, 1.3333)), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=(0.6, 1.4)), transforms.ToTensor(), transforms.Normalize(*norm)])