import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(21, 86, 118), padding_mode='constant'),
    transforms.RandomAffine(degrees=20, translate=(0.18, 0.19), scale=(0.89, 1.22), shear=(2.76, 7.7)),
    transforms.RandomGrayscale(p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
