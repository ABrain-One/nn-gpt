import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.82), ratio=(1.2, 2.17)),
    transforms.RandomPosterize(bits=4, p=0.24),
    transforms.RandomAffine(degrees=17, translate=(0.03, 0.08), scale=(0.86, 1.68), shear=(4.64, 9.37)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
