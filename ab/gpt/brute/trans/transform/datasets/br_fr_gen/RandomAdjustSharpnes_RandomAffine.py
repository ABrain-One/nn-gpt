import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.93, p=0.65),
    transforms.RandomAffine(degrees=9, translate=(0.02, 0.1), scale=(0.84, 1.63), shear=(4.75, 9.53)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
