import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.31), transforms.RandomAffine(degrees=17, translate=(0.1, 0.0), scale=(0.87, 1.73), shear=(4.14, 6.71)), transforms.RandomPerspective(distortion_scale=0.29, p=0.48), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])