import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomGrayscale(p=0.22), transforms.GaussianBlur(kernel_size=5, sigma=(0.79, 1.17)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])