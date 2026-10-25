import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.6, 0.8), ratio=(0.9, 2.5), antialias=True), transforms.RandomInvert(p=0.8), transforms.RandomGrayscale(p=0.3), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize([0.507, 0.486, 0.441], [0.267, 0.256, 0.276])])