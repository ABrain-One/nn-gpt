import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=11, translate=(0.12, 0.17), scale=(1.04, 1.6), shear=(3.62, 5.94)),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
