import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.56, p=0.86),
    transforms.RandomAffine(degrees=12, translate=(0.16, 0.05), scale=(1.16, 1.78), shear=(2.47, 8.16)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
