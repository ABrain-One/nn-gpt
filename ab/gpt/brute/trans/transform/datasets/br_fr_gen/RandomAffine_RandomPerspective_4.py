import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=11, translate=(0.06, 0.04), scale=(1.04, 1.7), shear=(3.75, 5.24)),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.85),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
