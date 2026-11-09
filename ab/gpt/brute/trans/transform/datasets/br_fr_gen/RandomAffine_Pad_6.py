import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.11, 0.09), scale=(1.17, 1.32), shear=(4.7, 5.54)),
    transforms.Pad(padding=1, fill=(142, 98, 39), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
