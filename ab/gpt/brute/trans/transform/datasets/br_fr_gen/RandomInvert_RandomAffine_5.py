import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.6),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.2), scale=(1.16, 1.68), shear=(1.94, 7.91)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
