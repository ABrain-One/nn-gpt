import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.RandomAutocontrast(p=0.81),
    transforms.RandomAffine(degrees=11, translate=(0.05, 0.07), scale=(0.83, 1.71), shear=(4.08, 8.55)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
