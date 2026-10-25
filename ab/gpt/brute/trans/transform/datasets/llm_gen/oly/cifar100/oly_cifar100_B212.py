import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.3), transforms.CenterCrop(30), transforms.RandomAdjustSharpness(sharpness_factor=0.73, p=0.87), transforms.RandomEqualize(p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])