import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=6, translate=(0.06, 0.15), scale=(0.96, 1.25), shear=(3.78, 9.67)),
    transforms.RandomVerticalFlip(p=0.57),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
