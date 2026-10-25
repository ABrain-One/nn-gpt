import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.3, p=0.15),
    transforms.RandomAffine(degrees=17, translate=(0.01, 0.13), scale=(0.9, 1.83), shear=(1.71, 7.54)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
