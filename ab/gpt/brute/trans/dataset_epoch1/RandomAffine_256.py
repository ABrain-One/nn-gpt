import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=24, translate=(0.13, 0.14), scale=(0.8, 1.58), shear=(2.38, 9.44)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
