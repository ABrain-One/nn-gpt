import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(22, 41, 234), padding_mode='edge'),
    transforms.RandomAffine(degrees=6, translate=(0.06, 0.18), scale=(1.18, 1.67), shear=(2.65, 8.76)),
    transforms.RandomRotation(degrees=6),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
