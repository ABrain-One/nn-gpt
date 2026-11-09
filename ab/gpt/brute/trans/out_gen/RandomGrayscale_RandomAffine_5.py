import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.41),
    transforms.RandomAffine(degrees=7, translate=(0.09, 0.14), scale=(0.88, 1.2), shear=(3.71, 7.01)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
