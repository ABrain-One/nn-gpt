import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.14),
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.94), ratio=(1.17, 1.57)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
