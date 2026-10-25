import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.96), ratio=(1.32, 2.55)),
    transforms.RandomAffine(degrees=25, translate=(0.18, 0.02), scale=(1.11, 1.35), shear=(2.02, 6.61)),
    transforms.CenterCrop(size=27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
