import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomGrayscale(p=0.76),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
