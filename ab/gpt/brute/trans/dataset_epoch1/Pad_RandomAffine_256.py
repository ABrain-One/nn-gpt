import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(33, 90, 155), padding_mode='edge'),
    transforms.RandomAffine(degrees=10, translate=(0.03, 0.05), scale=(1.08, 1.67), shear=(0.96, 5.4)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
