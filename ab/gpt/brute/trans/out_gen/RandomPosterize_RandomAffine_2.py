import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=4, p=0.32),
    transforms.RandomAffine(degrees=28, translate=(0.04, 0.02), scale=(1.16, 1.96), shear=(3.44, 5.38)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
