import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.12, 0.17), scale=(0.83, 1.61), shear=(1.14, 5.32)),
    transforms.Pad(padding=4, fill=(230, 96, 12), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
