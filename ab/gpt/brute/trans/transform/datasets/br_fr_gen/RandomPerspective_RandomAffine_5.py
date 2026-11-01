import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.14, p=0.33),
    transforms.RandomAffine(degrees=24, translate=(0.07, 0.09), scale=(1.19, 1.21), shear=(3.98, 6.05)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
