import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.89), ratio=(0.93, 2.16)),
    transforms.RandomGrayscale(p=0.62),
    transforms.RandomEqualize(p=0.54),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
