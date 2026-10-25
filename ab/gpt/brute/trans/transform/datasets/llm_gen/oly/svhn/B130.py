import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(degrees=(-10, 10)), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])