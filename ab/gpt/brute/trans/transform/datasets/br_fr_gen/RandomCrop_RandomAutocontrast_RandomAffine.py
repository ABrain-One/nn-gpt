import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.RandomAutocontrast(p=0.39),
    transforms.RandomAffine(degrees=15, translate=(0.02, 0.14), scale=(1.09, 1.28), shear=(3.48, 8.26)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
