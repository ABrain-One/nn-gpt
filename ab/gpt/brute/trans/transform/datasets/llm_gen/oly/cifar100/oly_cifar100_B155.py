import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.15), transforms.Pad(padding=4, fill=(164, 16, 92), padding_mode='constant'), transforms.Resize(224), transforms.ColorJitter(brightness=0.12, contrast=1.15, saturation=1.1, hue=0.07), transforms.RandomGrayscale(p=0.12), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(**norm)])