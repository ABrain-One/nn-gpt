import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.19, 0.12), scale=(1.01, 1.94), shear=(4.59, 7.26)),
    transforms.RandomHorizontalFlip(p=0.39),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
