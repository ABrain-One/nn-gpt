import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.19, 0.19), scale=(0.96, 1.78), shear=(3.49, 5.87)),
    transforms.RandomPosterize(bits=6, p=0.25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
