import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.15),
    transforms.Pad(padding=0, fill=(164, 16, 92), padding_mode='symmetric'),
    transforms.RandomGrayscale(p=0.12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
