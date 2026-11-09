import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.81),
    transforms.RandomAdjustSharpness(sharpness_factor=0.84, p=0.47),
    transforms.RandomAffine(degrees=23, translate=(0.12, 0.03), scale=(0.87, 1.28), shear=(3.77, 5.62)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
