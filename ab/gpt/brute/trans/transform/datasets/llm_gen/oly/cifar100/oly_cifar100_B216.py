import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(31), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.0, p=0.25), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAdjustSharpness(sharpness_factor=1.9, p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.486, 0.441], std=[0.267, 0.256, 0.276])])