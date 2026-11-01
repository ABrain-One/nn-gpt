import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=25, translate=(0.11, 0.01), scale=(0.83, 1.68), shear=(1.76, 7.02)),
    transforms.RandomInvert(p=0.49),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
