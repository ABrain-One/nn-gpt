import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(251, 154, 46), padding_mode='reflect'), transforms.RandomAffine(degrees=10, translate=(0.03, 0.08), scale=(1.07, 1.57), shear=(1.27, 5.88)), transforms.RandomAutocontrast(p=0.11), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])