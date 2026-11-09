import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.Pad(padding=5, fill=(221, 178, 245), padding_mode='constant'),
    transforms.RandomRotation(degrees=4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
