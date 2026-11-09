import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=3, fill=(46, 111, 107), padding_mode='constant'),
        transforms.RandomRotation(degrees=1),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])