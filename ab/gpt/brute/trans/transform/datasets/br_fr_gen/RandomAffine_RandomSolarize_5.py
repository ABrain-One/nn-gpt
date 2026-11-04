import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=1, translate=(0.16, 0.13), scale=(0.88, 1.55), shear=(4.07, 6.55)),
    transforms.RandomSolarize(threshold=87, p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
