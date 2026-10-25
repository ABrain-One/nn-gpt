import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=240, p=0.48),
    transforms.RandomAffine(degrees=12, translate=(0.16, 0.01), scale=(0.81, 1.35), shear=(3.64, 5.1)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
