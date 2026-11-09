import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.16, p=0.21),
    transforms.RandomAffine(degrees=16, translate=(0.17, 0.16), scale=(0.81, 1.83), shear=(4.12, 5.21)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
