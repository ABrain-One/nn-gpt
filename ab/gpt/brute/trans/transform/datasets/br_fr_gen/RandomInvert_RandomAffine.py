import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.22),
    transforms.RandomAffine(degrees=3, translate=(0.11, 0.12), scale=(1.03, 1.72), shear=(1.91, 6.81)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
