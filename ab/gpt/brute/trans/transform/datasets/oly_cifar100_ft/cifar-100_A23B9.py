import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.RandomGrayscale(p=0.67), transforms.GaussianBlur(kernel_size=5, sigma=(0.63, 1.52)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])