import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(135, 11, 242), padding_mode='symmetric'), transforms.RandomRotation(degrees=12), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])