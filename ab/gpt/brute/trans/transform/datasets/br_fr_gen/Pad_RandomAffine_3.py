import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(138, 10, 173), padding_mode='edge'),
    transforms.RandomAffine(degrees=16, translate=(0.05, 0.16), scale=(0.88, 1.2), shear=(1.07, 9.66)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
