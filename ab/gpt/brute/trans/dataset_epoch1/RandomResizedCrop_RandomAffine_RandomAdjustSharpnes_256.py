import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.99), ratio=(1.02, 2.44)),
    transforms.RandomAffine(degrees=3, translate=(0.08, 0.17), scale=(1.04, 1.37), shear=(1.95, 9.21)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.79, p=0.21),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
