import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.79),
    transforms.RandomPerspective(distortion_scale=0.11, p=0.17),
    transforms.RandomAffine(degrees=23, translate=(0.18, 0.14), scale=(0.95, 1.35), shear=(0.36, 8.84)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
