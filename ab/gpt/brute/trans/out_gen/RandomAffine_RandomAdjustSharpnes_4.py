import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=13, translate=(0.03, 0.09), scale=(0.87, 1.27), shear=(1.66, 7.55)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.73, p=0.25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
