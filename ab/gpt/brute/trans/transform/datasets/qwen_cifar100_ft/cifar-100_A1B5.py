import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(72, 164, 130), padding_mode='reflect'), transforms.RandomHorizontalFlip(p=0.59), transforms.RandomAutocontrast(p=0.22), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])