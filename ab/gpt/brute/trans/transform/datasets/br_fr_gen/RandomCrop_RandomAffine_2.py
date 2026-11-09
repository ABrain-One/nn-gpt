import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomAffine(degrees=12, translate=(0.03, 0.12), scale=(1.08, 1.95), shear=(0.6, 6.51)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
