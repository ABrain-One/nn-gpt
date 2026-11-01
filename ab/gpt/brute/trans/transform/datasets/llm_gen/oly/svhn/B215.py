import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(63, 38, 167), padding_mode='constant'), transforms.RandomVerticalFlip(p=0.2), transforms.Resize((64, 64)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0), transforms.ToTensor(), transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.2675, 0.2565, 0.2762])])