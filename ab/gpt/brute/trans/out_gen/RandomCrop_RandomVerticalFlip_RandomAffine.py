import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.RandomVerticalFlip(p=0.54),
    transforms.RandomAffine(degrees=21, translate=(0.11, 0.16), scale=(1.04, 1.33), shear=(2.43, 6.72)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
