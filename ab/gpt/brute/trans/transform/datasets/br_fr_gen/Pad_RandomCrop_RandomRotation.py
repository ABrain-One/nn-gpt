import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(176, 195, 103), padding_mode='reflect'),
    transforms.RandomCrop(size=32),
    transforms.RandomRotation(degrees=16),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
