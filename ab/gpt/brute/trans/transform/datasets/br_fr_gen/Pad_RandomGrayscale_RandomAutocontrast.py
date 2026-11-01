import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(109, 226, 156), padding_mode='symmetric'),
    transforms.RandomGrayscale(p=0.86),
    transforms.RandomAutocontrast(p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
