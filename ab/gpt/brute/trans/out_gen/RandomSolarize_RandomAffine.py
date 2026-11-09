import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=231, p=0.79),
    transforms.RandomAffine(degrees=25, translate=(0.07, 0.12), scale=(1.19, 1.21), shear=(0.38, 6.13)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
