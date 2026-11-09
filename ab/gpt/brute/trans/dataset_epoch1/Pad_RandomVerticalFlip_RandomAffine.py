import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(88, 13, 95), padding_mode='constant'),
    transforms.RandomVerticalFlip(p=0.63),
    transforms.RandomAffine(degrees=2, translate=(0.07, 0.18), scale=(0.88, 1.68), shear=(4.9, 6.43)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
