import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.RandomPerspective(distortion_scale=0.2, p=0.3), transforms.ColorJitter(brightness=0.81, contrast=1.02, saturation=1.0, hue=0.02), transforms.GaussianBlur(kernel_size=(3, 3), sigma=1.0, p=0.5), transforms.Resize(224, interpolation=transforms.InterpolationMode.LANCZOS), transforms.ToTensor(), transforms.Normalize(*norm)])