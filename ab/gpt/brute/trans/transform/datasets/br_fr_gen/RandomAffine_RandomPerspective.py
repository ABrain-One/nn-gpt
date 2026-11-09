import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=23, translate=(0.06, 0.07), scale=(1.11, 1.21), shear=(1.72, 6.47)),
    transforms.RandomPerspective(distortion_scale=0.13, p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
