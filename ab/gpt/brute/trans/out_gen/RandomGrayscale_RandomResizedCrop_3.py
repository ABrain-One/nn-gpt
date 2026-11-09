import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.92), ratio=(1.16, 1.69)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
