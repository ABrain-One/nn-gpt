import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.RandomAffine(degrees=7, translate=(0.09, 0.09), scale=(1.06, 1.82), shear=(0.85, 7.04)),
    transforms.RandomInvert(p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
