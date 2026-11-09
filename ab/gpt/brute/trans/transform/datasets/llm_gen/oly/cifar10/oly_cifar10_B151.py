import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(102, 234, 199), padding_mode='constant'), transforms.RandomCrop(32), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(10), transforms.Resize((64, 64)), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.ToTensor(), transforms.Normalize(*norm)])