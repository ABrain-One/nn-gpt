import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(67, 59, 252), padding_mode='symmetric'),
    transforms.RandomGrayscale(p=0.43),
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.97), ratio=(0.95, 2.72)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
