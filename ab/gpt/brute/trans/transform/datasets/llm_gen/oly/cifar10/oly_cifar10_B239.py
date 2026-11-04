import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8), transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(kernel_size=3), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])