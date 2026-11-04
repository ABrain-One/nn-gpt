import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=7, p=0.53),
    transforms.RandomAffine(degrees=27, translate=(0.09, 0.12), scale=(0.9, 1.4), shear=(4.28, 5.23)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
