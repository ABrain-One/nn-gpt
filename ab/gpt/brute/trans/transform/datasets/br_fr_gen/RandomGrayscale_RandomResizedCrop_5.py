import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.86),
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.93), ratio=(0.81, 2.73)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
