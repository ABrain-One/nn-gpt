import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.RandomPosterize(bits=4, p=0.26), transforms.GaussianBlur(kernel_size=3, sigma=(0.87, 1.14)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])