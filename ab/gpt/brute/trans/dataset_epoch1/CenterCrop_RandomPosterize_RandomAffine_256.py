import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomPosterize(bits=5, p=0.64),
    transforms.RandomAffine(degrees=24, translate=(0.15, 0.01), scale=(0.82, 1.48), shear=(1.01, 9.39)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
