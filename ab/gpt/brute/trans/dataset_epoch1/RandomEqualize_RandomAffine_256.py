import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.74),
    transforms.RandomAffine(degrees=1, translate=(0.14, 0.14), scale=(0.9, 1.36), shear=(3.01, 9.74)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
