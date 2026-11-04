import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(157, 48, 62), padding_mode='constant'),
    transforms.RandomEqualize(p=0.1),
    transforms.RandomAffine(degrees=5, translate=(0.01, 0.18), scale=(1.04, 1.53), shear=(4.32, 9.05)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
