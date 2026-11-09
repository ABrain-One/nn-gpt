import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.32),
    transforms.RandomAffine(degrees=7, translate=(0.06, 0.04), scale=(1.03, 1.73), shear=(3.04, 7.41)),
    transforms.RandomHorizontalFlip(p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
