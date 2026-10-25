import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=16, translate=(0.07, 0.11), scale=(0.9, 1.74), shear=(1.38, 8.75)),
    transforms.RandomSolarize(threshold=242, p=0.59),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
