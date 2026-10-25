import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomAdjustSharpness(sharpness_factor=1.29, p=0.59),
    transforms.RandomAffine(degrees=10, translate=(0.2, 0.06), scale=(0.81, 1.71), shear=(3.69, 5.26)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
