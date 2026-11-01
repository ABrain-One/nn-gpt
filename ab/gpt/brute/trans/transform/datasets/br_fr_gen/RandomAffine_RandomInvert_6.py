import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=21, translate=(0.05, 0.18), scale=(1.05, 1.64), shear=(2.64, 9.12)),
    transforms.RandomInvert(p=0.63),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
