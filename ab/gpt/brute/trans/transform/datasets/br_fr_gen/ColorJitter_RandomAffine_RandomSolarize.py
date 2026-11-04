import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.05, contrast=0.92, saturation=1.18, hue=0.05),
    transforms.RandomAffine(degrees=22, translate=(0.03, 0.07), scale=(0.85, 1.26), shear=(1.82, 9.38)),
    transforms.RandomSolarize(threshold=248, p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
