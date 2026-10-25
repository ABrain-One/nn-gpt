import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomAffine(degrees=26, translate=(0.14, 0.07), scale=(1.15, 1.78), shear=(3.39, 6.76)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.69, p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
