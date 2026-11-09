import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.RandomGrayscale(p=0.82), transforms.GaussianBlur(kernel_size=3, sigma=(0.39, 1.58)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])