import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomPosterize(bits=4, p=0.58),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.19), scale=(1.07, 1.35), shear=(4.18, 6.14)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
