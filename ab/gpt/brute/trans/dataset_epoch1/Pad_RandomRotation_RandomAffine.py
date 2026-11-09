import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(111, 227, 54), padding_mode='edge'),
    transforms.RandomRotation(degrees=21),
    transforms.RandomAffine(degrees=2, translate=(0.09, 0.11), scale=(0.89, 1.27), shear=(1.75, 6.18)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
