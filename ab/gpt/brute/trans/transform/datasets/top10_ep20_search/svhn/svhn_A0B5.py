import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(77, 187, 240), padding_mode='constant'), transforms.RandomVerticalFlip(p=0.7), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2), transforms.RandomRotation(degrees=22), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])