import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.89),
    transforms.RandomAffine(degrees=4, translate=(0.2, 0.07), scale=(1.15, 1.71), shear=(3.67, 8.44)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
