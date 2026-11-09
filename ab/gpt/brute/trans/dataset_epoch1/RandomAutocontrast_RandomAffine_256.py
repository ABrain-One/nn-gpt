import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.32),
    transforms.RandomAffine(degrees=27, translate=(0.07, 0.09), scale=(1.05, 1.72), shear=(3.73, 7.54)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
