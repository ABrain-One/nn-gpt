import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.31),
    transforms.RandomGrayscale(p=0.39),
    transforms.ColorJitter(brightness=0.99, contrast=1.17, saturation=0.98, hue=0.05),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
