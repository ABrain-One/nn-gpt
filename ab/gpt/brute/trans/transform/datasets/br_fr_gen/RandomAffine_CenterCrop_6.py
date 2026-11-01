import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=17, translate=(0.01, 0.08), scale=(0.81, 1.9), shear=(3.42, 5.74)),
    transforms.CenterCrop(size=26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
