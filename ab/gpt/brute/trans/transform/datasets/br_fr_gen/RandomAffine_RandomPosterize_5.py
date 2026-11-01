import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=29, translate=(0.18, 0.1), scale=(1.15, 1.46), shear=(3.41, 5.94)),
    transforms.RandomPosterize(bits=8, p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
