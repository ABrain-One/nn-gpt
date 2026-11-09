import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.RandomAffine(degrees=8, translate=(0.16, 0.11), scale=(1.05, 1.43), shear=(0.21, 5.03)),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.96), ratio=(1.22, 2.51)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
