import torch
import torchvision.transforms as transforms
from torchvision import functional as F

def transform(norm):
    return transforms.Compose([transforms.Resize(40, interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.Lambda(lambda x: F.gaussian_blur(x, kernel_size=3, sigma=0.5))], p=0.5), transforms.RandomApply([transforms.Lambda(lambda x: F.posterize(x, 5))], p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])