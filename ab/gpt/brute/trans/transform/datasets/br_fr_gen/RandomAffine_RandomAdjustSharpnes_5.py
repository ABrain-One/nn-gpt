import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=26, translate=(0.19, 0.06), scale=(0.86, 1.33), shear=(2.66, 7.93)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.79, p=0.42),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
