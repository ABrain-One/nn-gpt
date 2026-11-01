import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=2, translate=(0.12, 0.19), scale=(0.86, 1.7), shear=(4.77, 5.27)),
    transforms.RandomEqualize(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
