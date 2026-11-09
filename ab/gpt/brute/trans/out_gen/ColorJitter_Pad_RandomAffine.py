import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.01, contrast=0.96, saturation=1.04, hue=0.04),
    transforms.Pad(padding=5, fill=(102, 57, 227), padding_mode='constant'),
    transforms.RandomAffine(degrees=16, translate=(0.07, 0.05), scale=(0.84, 1.58), shear=(4.64, 7.48)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
