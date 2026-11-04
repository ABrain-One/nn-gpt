import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5), transforms.RandomGrayscale(p=0.1), transforms.Pad(padding=4, fill=(126, 104, 88), padding_mode='edge'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])