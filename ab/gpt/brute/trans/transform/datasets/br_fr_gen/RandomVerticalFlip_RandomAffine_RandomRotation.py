import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.34),
    transforms.RandomAffine(degrees=19, translate=(0.11, 0.14), scale=(0.81, 1.44), shear=(4.83, 7.41)),
    transforms.RandomRotation(degrees=18),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
