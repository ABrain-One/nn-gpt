import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomInvert(p=0.66),
    transforms.RandomAffine(degrees=22, translate=(0.02, 0.16), scale=(0.98, 1.62), shear=(2.95, 5.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
