import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.88), ratio=(1.18, 2.76)),
    transforms.RandomHorizontalFlip(p=0.61),
    transforms.RandomGrayscale(p=0.26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
