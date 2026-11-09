import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=Image.BICUBIC), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])