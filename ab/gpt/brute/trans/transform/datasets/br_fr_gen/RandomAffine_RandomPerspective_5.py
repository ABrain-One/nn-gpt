import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=11, translate=(0.09, 0.09), scale=(0.94, 1.9), shear=(3.99, 5.87)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
