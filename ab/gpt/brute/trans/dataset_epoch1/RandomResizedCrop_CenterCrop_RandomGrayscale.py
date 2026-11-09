import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.81), ratio=(0.85, 2.07)),
    transforms.CenterCrop(size=24),
    transforms.RandomGrayscale(p=0.39),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
