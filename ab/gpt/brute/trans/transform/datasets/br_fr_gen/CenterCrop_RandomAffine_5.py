import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.RandomAffine(degrees=23, translate=(0.02, 0.09), scale=(0.87, 1.43), shear=(3.85, 6.16)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
