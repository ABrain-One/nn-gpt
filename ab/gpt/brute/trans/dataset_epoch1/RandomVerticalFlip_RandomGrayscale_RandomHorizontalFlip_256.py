import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.27),
    transforms.RandomGrayscale(p=0.59),
    transforms.RandomHorizontalFlip(p=0.87),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
