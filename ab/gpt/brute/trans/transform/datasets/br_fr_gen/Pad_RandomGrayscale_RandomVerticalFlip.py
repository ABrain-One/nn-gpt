import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(35, 247, 182), padding_mode='symmetric'),
    transforms.RandomGrayscale(p=0.24),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
