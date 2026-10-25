import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=27), transforms.GaussianBlur(kernel_size=3, sigma=(0.94, 1.24)), transforms.RandomVerticalFlip(p=0.83), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])