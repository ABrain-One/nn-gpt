import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.68),
    transforms.RandomAffine(degrees=11, translate=(0.15, 0.12), scale=(0.88, 1.62), shear=(4.81, 5.04)),
    transforms.RandomPosterize(bits=8, p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
