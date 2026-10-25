import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.95, contrast=0.83, saturation=1.15, hue=0.09),
    transforms.RandomAffine(degrees=16, translate=(0.03, 0.05), scale=(0.87, 1.25), shear=(0.62, 6.15)),
    transforms.Pad(padding=1, fill=(133, 129, 160), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
