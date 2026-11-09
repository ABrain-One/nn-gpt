import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.Pad(padding=3, fill=(172, 150, 24), padding_mode='reflect'),
    transforms.RandomAffine(degrees=6, translate=(0.19, 0.19), scale=(0.87, 1.51), shear=(0.26, 8.84)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
