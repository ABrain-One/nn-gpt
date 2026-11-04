import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, padding_mode='symmetric'), transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), shear=(-5, 5), p=0.8), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8), transforms.GaussianBlur(kernel_size=3, sigma=1.5, p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=norm[0], std=norm[1], inplace=True)])