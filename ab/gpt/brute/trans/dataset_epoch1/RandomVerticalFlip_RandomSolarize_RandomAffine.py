import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.43),
    transforms.RandomSolarize(threshold=153, p=0.8),
    transforms.RandomAffine(degrees=11, translate=(0.15, 0.04), scale=(1.03, 1.49), shear=(3.37, 5.9)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
