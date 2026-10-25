import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomAffine(degrees=26, translate=(0.15, 0.1), scale=(0.82, 1.82), shear=(4.59, 5.97)),
    transforms.RandomPerspective(distortion_scale=0.21, p=0.1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
