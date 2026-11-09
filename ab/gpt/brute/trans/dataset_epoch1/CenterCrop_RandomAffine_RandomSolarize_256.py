import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomAffine(degrees=21, translate=(0.15, 0.18), scale=(0.81, 1.39), shear=(3.46, 5.2)),
    transforms.RandomSolarize(threshold=190, p=0.4),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
