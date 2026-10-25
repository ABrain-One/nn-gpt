import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.GaussianBlur(kernel_size=3, sigma=(0.96, 1.57)), transforms.RandomHorizontalFlip(p=0.16), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])