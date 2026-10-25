import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=16, translate=(0.15, 0.19), scale=(1.09, 1.84), shear=(4.63, 7.9)),
    transforms.RandomSolarize(threshold=138, p=0.47),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
