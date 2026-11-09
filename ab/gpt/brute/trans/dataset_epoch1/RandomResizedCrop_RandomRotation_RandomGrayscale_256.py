import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.89), ratio=(1.05, 2.69)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomGrayscale(p=0.8),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
