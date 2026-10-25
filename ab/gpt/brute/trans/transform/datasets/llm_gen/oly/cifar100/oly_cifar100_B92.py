import torch
import torchvision.transforms as transforms
from PIL import Image

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomApply([transforms.AutoContrast()], p=0.5), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.25), shear=(-5.0, 5.0), resample=Image.BICUBIC, fillcolor=(22, 95, 60)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.5)], p=0.5), transforms.RandomGrayscale(p=0.2), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(**norm)])