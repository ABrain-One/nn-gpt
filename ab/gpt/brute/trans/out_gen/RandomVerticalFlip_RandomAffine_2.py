import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.77),
    transforms.RandomAffine(degrees=24, translate=(0.2, 0.04), scale=(1.19, 1.31), shear=(2.64, 6.07)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
