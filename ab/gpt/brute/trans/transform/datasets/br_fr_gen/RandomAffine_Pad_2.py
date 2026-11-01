import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.11, 0.12), scale=(0.91, 1.74), shear=(0.67, 9.42)),
    transforms.Pad(padding=3, fill=(222, 157, 4), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
