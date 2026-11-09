import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(42, 103, 19), padding_mode='constant'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.17, p=0.35),
    transforms.RandomAffine(degrees=27, translate=(0.13, 0.11), scale=(1.01, 2.0), shear=(4.19, 8.65)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
