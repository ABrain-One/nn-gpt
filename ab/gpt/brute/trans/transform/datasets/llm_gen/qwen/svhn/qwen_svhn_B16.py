import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.8, 0.95), ratio=(0.75, 1.25)), transforms.Pad(padding=2, fill=(220, 185, 72), padding_mode='symmetric'), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomGrayscale(p=0.1), transforms.RandomEqualize(p=0.4), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])