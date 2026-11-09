import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.33),
    transforms.RandomAffine(degrees=28, translate=(0.09, 0.07), scale=(1.09, 1.84), shear=(2.07, 6.93)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
