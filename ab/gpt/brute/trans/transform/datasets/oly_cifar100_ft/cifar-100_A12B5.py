import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(168, 158, 64), padding_mode='reflect'), transforms.RandomRotation(degrees=10), transforms.RandomSolarize(threshold=100, p=0.43), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])