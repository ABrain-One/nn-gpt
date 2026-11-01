import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=184, p=0.27),
    transforms.RandomAffine(degrees=22, translate=(0.17, 0.13), scale=(0.82, 1.43), shear=(2.16, 9.58)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
