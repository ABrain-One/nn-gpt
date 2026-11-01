import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.23),
    transforms.RandomAffine(degrees=16, translate=(0.06, 0.02), scale=(1.11, 1.59), shear=(4.43, 6.46)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
