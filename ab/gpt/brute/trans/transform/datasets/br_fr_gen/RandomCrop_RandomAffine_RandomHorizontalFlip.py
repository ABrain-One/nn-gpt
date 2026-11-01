import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.RandomAffine(degrees=12, translate=(0.09, 0.07), scale=(0.96, 1.23), shear=(1.01, 5.75)),
    transforms.RandomHorizontalFlip(p=0.23),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
