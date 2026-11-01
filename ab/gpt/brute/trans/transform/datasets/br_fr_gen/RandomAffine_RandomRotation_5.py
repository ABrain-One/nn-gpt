import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=9, translate=(0.15, 0.12), scale=(0.91, 1.32), shear=(0.75, 9.69)),
    transforms.RandomRotation(degrees=5),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
