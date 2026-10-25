import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.7), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)), transforms.Grayscale(num_output_channels=3), transforms.Resize((64, 64)), transforms.RandomAffine(15, scale=(0.8, 1.2), translate=(0.1, 0.1)), transforms.ToTensor(), transforms.Normalize(*norm)])