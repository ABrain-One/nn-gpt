import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(167, 167, 241), padding_mode='constant'), transforms.ColorJitter(brightness=0.84, contrast=1.1, saturation=0.85, hue=0.07), transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.39), transforms.RandomRotation(degrees=13), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])