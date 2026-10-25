import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.15),
    transforms.RandomAffine(degrees=23, translate=(0.13, 0.16), scale=(1.2, 1.28), shear=(4.24, 6.02)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
