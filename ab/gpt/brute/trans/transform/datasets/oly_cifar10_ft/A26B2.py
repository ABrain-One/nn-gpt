import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(234, 89, 157), padding_mode='constant'),
    transforms.RandomRotation(degrees=1),
    transforms.RandomAutocontrast(p=0.72),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])