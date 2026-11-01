import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=2, translate=(0.02, 0.06), scale=(1.17, 1.71), shear=(3.53, 8.26)),
    transforms.RandomGrayscale(p=0.46),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
