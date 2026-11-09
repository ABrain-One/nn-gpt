import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.84), ratio=(0.91, 2.36)),
    transforms.RandomAffine(degrees=15, translate=(0.04, 0.19), scale=(1.18, 1.65), shear=(4.43, 5.23)),
    transforms.RandomPosterize(bits=4, p=0.34),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
