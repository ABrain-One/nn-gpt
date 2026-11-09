import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.52),
    transforms.RandomEqualize(p=0.46),
    transforms.RandomAffine(degrees=5, translate=(0.15, 0.02), scale=(0.97, 1.53), shear=(0.25, 5.77)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
