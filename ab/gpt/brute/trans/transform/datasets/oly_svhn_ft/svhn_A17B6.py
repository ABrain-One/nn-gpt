import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.72, 0.9), ratio=(1.11, 1.71)), transforms.RandomGrayscale(p=0.16), transforms.Pad(padding=0, fill=(145, 147, 211), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])