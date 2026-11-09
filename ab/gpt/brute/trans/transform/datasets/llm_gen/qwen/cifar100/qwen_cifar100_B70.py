import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.65, 0.87), ratio=(1.32, 2.17)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGrayscale(p=0.2), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5), transforms.Pad(padding=1, fill=(128, 128, 128), padding_mode='constant'), transforms.RandomCrop(size=32), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])