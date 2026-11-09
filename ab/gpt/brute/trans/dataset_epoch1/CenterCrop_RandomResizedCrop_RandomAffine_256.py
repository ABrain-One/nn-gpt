import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.91), ratio=(1.09, 1.55)),
    transforms.RandomAffine(degrees=25, translate=(0.17, 0.07), scale=(1.04, 1.42), shear=(0.39, 5.05)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
