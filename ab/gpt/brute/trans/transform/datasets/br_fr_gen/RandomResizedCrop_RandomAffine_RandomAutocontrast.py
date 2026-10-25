import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.97), ratio=(1.32, 2.44)),
    transforms.RandomAffine(degrees=25, translate=(0.06, 0.18), scale=(1.03, 1.89), shear=(1.38, 9.61)),
    transforms.RandomAutocontrast(p=0.77),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
