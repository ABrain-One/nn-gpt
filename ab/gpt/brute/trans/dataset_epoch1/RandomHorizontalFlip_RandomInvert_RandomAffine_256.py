import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.46),
    transforms.RandomInvert(p=0.8),
    transforms.RandomAffine(degrees=5, translate=(0.11, 0.01), scale=(0.83, 1.97), shear=(4.17, 6.62)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
