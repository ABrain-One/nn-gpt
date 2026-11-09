import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomRotation(degrees=18),
    transforms.RandomAffine(degrees=20, translate=(0.07, 0.17), scale=(0.98, 1.35), shear=(0.56, 5.72)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
