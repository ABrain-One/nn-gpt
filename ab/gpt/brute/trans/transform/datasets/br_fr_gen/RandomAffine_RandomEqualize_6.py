import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=24, translate=(0.2, 0.15), scale=(0.92, 1.36), shear=(0.64, 6.34)),
    transforms.RandomEqualize(p=0.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
