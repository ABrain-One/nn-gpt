import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.76), transforms.GaussianBlur(kernel_size=3, sigma=(0.21, 1.31)), transforms.RandomPosterize(bits=4, p=0.88), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])