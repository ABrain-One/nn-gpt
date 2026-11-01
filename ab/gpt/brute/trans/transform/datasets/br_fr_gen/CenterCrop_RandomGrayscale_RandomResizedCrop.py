import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.RandomGrayscale(p=0.68),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.97), ratio=(1.33, 1.88)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
