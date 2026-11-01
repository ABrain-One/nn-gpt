import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.89),
    transforms.RandomAffine(degrees=3, translate=(0.16, 0.14), scale=(0.86, 1.95), shear=(4.18, 5.46)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
