import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=24), transforms.GaussianBlur(kernel_size=5, sigma=(0.9, 1.31)), transforms.RandomSolarize(threshold=190, p=0.76), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])