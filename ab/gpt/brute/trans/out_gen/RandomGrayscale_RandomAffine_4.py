import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.39),
    transforms.RandomAffine(degrees=7, translate=(0.01, 0.06), scale=(1.11, 1.9), shear=(3.58, 5.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
