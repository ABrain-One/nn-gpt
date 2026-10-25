import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1), transforms.Grayscale(num_output_channels=3), transforms.RandomVerticalFlip(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])