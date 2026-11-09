import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomAffine(degrees=29, translate=(0.16, 0.16), scale=(1.14, 1.32), shear=(3.6, 6.75)),
    transforms.RandomInvert(p=0.26),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
