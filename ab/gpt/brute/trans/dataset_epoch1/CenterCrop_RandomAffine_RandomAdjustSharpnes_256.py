import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.RandomAffine(degrees=19, translate=(0.13, 0.03), scale=(1.18, 1.55), shear=(2.11, 7.47)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.86, p=0.62),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
