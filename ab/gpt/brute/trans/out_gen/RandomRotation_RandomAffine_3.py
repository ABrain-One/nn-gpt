import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=29),
    transforms.RandomAffine(degrees=23, translate=(0.16, 0.07), scale=(1.07, 1.27), shear=(4.86, 9.57)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
