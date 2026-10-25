import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=12, translate=(0.16, 0.08), scale=(0.89, 1.31), shear=(4.8, 6.22)),
    transforms.RandomPosterize(bits=7, p=0.27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
