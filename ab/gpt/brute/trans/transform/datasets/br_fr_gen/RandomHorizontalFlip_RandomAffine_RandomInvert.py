import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomAffine(degrees=15, translate=(0.17, 0.15), scale=(1.12, 1.25), shear=(1.65, 9.45)),
    transforms.RandomInvert(p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
