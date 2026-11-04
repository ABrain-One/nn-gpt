import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.19),
    transforms.RandomAdjustSharpness(sharpness_factor=1.06, p=0.5),
    transforms.RandomAffine(degrees=18, translate=(0.16, 0.05), scale=(0.82, 1.98), shear=(2.09, 7.45)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
