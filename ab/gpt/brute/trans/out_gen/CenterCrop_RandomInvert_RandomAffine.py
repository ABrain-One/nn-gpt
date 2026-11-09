import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomInvert(p=0.57),
    transforms.RandomAffine(degrees=14, translate=(0.08, 0.2), scale=(0.85, 1.32), shear=(1.51, 8.09)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
