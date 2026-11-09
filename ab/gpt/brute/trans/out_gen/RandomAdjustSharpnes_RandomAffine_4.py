import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=0.55, p=0.46),
    transforms.RandomAffine(degrees=20, translate=(0.12, 0.04), scale=(1.03, 1.48), shear=(0.93, 7.15)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
