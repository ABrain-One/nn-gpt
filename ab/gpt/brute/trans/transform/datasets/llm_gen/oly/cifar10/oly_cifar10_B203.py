import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomGrayscale(p=0.27), transforms.RandomHorizontalFlip(p=0.5), transforms.Pad(padding=2, fill=(123, 38, 57), padding_mode='symmetric'), transforms.RandomRotation(degrees=15), transforms.Resize((64, 64)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ToTensor(), transforms.Normalize(*norm)])