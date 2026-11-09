import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.83), ratio=(1.09, 2.95)),
    transforms.RandomAffine(degrees=15, translate=(0.17, 0.0), scale=(1.0, 1.59), shear=(3.62, 8.66)),
    transforms.Pad(padding=2, fill=(146, 195, 239), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
