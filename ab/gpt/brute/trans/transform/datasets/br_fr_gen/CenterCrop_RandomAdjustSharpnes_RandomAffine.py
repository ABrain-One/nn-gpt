import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.RandomAdjustSharpness(sharpness_factor=1.85, p=0.49),
    transforms.RandomAffine(degrees=7, translate=(0.02, 0.1), scale=(1.17, 1.63), shear=(4.47, 6.38)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
