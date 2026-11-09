import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAutocontrast(p=0.35), transforms.Pad(padding=3, fill=(234, 244, 172), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])