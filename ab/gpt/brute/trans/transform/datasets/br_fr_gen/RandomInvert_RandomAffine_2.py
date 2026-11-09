import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.59),
    transforms.RandomAffine(degrees=29, translate=(0.15, 0.11), scale=(0.95, 1.72), shear=(2.88, 5.15)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
