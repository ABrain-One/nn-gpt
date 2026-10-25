import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.3), transforms.RandomAffine(degrees=14, translate=(0.03, 0.03), scale=(1.01, 1.62), shear=(1.98, 9.97)), transforms.Pad(padding=3, fill=(137, 75, 184), padding_mode='symmetric'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])