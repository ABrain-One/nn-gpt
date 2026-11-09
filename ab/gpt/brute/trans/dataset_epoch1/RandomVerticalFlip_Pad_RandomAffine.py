import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.15),
    transforms.Pad(padding=3, fill=(64, 5, 248), padding_mode='constant'),
    transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(1.05, 1.86), shear=(1.1, 5.35)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
