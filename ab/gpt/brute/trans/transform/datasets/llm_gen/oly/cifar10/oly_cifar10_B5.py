import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), transforms.RandomGrayscale(p=0.5), transforms.RandomAdjustSharpness(sharpness_factor=0.69, p=0.75), transforms.RandomRotation(degrees=15), transforms.GaussianBlur(kernel_size=3, sigma=1.5, p=0.5), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(**norm)])