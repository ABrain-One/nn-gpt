import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=17),
    transforms.RandomAffine(degrees=18, translate=(0.09, 0.16), scale=(0.88, 1.29), shear=(1.15, 5.72)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
