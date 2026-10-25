import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(217, 14, 227), padding_mode='constant'), transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.99, 1.01)), transforms.ColorJitter(brightness=1.08, contrast=1.11, saturation=0.98, hue=0.04), transforms.RandomAdjustSharpness(sharpness_factor=1.17, p=0.57), transforms.RandomAffine(degrees=20, translate=(0.18, 0.12), scale=(1.17, 1.49), shear=(4.61, 7.17)), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(*norm)])