import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.92), ratio=(1.3, 2.78)),
    transforms.RandomSolarize(threshold=81, p=0.48),
    transforms.RandomAffine(degrees=29, translate=(0.05, 0.07), scale=(1.14, 1.33), shear=(0.26, 5.88)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
