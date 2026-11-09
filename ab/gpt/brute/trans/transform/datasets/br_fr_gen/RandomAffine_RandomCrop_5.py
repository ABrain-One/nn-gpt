import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=9, translate=(0.04, 0.11), scale=(1.0, 1.47), shear=(3.76, 9.44)),
    transforms.RandomCrop(size=29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
