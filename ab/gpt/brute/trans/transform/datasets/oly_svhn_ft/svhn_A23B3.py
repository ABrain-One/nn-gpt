import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAffine(degrees=3, translate=(0.03, 0.06), scale=(1.09, 1.22), shear=(3.69, 5.73)), transforms.RandomGrayscale(p=0.45), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])