import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, padding_mode='reflect'), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=0.5), transforms.RandomRotation(degrees=(-10, 10), interpolate=transforms.InterpolationMode.BICUBIC), transforms.RandomPerspective(distortion_scale=0.05, p=0.5), transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])