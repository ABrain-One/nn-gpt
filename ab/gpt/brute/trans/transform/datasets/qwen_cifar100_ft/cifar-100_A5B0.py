import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.18), transforms.GaussianBlur(kernel_size=5, sigma=(0.62, 1.42)), transforms.RandomVerticalFlip(p=0.33), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])