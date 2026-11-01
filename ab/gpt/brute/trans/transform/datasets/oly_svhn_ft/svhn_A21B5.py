import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(105, 18, 242), padding_mode='reflect'), transforms.RandomAutocontrast(p=0.76), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])