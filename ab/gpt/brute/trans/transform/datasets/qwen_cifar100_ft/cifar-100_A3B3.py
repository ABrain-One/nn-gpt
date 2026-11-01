import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=8), transforms.Pad(padding=3, fill=(71, 253, 229), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])