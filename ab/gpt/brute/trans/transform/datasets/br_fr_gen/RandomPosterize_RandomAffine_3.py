import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=8, p=0.26),
    transforms.RandomAffine(degrees=24, translate=(0.09, 0.14), scale=(0.81, 1.5), shear=(0.24, 5.28)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
