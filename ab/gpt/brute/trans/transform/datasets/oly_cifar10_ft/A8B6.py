import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(221, 98, 117), padding_mode='constant'),
    transforms.RandomAutocontrast(p=0.74),
    transforms.RandomAffine(degrees=11, translate=(0.05, 0.09), scale=(1.04, 1.91), shear=(3.76, 6.14)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])