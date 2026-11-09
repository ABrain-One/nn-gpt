import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=18, translate=(0.19, 0.16), scale=(0.99, 1.45), shear=(0.89, 8.77)),
    transforms.RandomEqualize(p=0.85),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
