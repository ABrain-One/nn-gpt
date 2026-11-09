import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.41),
    transforms.RandomGrayscale(p=0.27),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.9), ratio=(1.26, 1.51)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
