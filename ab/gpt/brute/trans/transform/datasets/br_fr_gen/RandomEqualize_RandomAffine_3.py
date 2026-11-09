import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.76),
    transforms.RandomAffine(degrees=3, translate=(0.15, 0.03), scale=(0.9, 1.82), shear=(3.58, 8.44)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
