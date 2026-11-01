import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(147, 214, 96), padding_mode='constant'),
    transforms.RandomAffine(degrees=26, translate=(0.16, 0.11), scale=(1.14, 1.81), shear=(2.74, 7.73)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
