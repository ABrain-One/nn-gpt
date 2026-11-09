import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomAffine(degrees=28, translate=(0.08, 0.09), scale=(0.91, 1.96), shear=(1.52, 5.74)),
    transforms.RandomPosterize(bits=6, p=0.16),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
