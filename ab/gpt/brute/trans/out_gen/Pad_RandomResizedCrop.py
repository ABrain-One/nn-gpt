import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(124, 205, 212), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.9), ratio=(1.28, 2.37)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
