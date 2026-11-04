import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=22, translate=(0.01, 0.18), scale=(1.16, 1.25), shear=(1.44, 6.42)),
    transforms.RandomPosterize(bits=6, p=0.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
