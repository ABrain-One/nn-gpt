import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=12, translate=(0.02, 0.15), scale=(0.9, 1.66), shear=(0.6, 9.25)),
    transforms.Pad(padding=2, fill=(242, 150, 6), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
