import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomAffine(degrees=11, translate=(0.15, 0.18), scale=(1.07, 1.38), shear=(0.79, 5.63)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
