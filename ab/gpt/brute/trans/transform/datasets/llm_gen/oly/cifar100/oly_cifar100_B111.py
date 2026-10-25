import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(105, 8, 169), padding_mode='symmetric'), transforms.Grayscale(num_output_channels=3), transforms.RandomAffine(degrees=23, translate=(0.2, 0.25), scale=(0.9, 1.5), shear=(5, 8)), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1), transforms.RandomHorizontalFlip(p=0.59), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])