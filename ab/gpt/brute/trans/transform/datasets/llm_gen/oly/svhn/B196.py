import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(27), transforms.RandomEqualize(p=0.3), transforms.GaussianBlur(kernel_size=3, sigma=(0.8, 1.6)), transforms.RandomVerticalFlip(p=0.2), transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])