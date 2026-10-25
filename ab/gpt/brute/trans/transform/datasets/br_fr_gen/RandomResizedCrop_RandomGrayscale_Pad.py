import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.96), ratio=(0.91, 2.47)),
    transforms.RandomGrayscale(p=0.32),
    transforms.Pad(padding=3, fill=(60, 22, 181), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
