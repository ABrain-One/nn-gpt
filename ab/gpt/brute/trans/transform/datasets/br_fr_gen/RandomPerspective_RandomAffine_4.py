import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.12, p=0.65),
    transforms.RandomAffine(degrees=24, translate=(0.01, 0.04), scale=(1.01, 1.91), shear=(3.04, 7.1)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
