import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomAffine(degrees=1, translate=(0.13, 0.05), scale=(1.07, 1.46), shear=(1.15, 5.6)),
    transforms.RandomVerticalFlip(p=0.44),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
