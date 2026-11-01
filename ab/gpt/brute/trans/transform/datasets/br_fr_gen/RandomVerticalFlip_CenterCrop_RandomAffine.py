import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.28),
    transforms.CenterCrop(size=26),
    transforms.RandomAffine(degrees=5, translate=(0.17, 0.16), scale=(0.98, 1.55), shear=(3.32, 6.07)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
