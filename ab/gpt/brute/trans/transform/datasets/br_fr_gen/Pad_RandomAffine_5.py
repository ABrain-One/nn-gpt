import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(88, 47, 40), padding_mode='edge'),
    transforms.RandomAffine(degrees=6, translate=(0.15, 0.15), scale=(0.88, 1.32), shear=(2.05, 6.97)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
