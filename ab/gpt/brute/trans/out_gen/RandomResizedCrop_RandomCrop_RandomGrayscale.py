import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.9), ratio=(1.25, 2.84)),
    transforms.RandomCrop(size=25),
    transforms.RandomGrayscale(p=0.42),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
