import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=25), transforms.RandomPosterize(bits=4, p=0.78), transforms.GaussianBlur(kernel_size=5, sigma=(0.85, 1.55)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])