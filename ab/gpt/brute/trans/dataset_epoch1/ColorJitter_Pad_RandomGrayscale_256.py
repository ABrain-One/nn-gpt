import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.89, contrast=0.97, saturation=0.92, hue=0.05),
    transforms.Pad(padding=3, fill=(132, 58, 218), padding_mode='symmetric'),
    transforms.RandomGrayscale(p=0.16),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
