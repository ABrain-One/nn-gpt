import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(191, 87, 117), padding_mode='symmetric'),
    transforms.RandomAutocontrast(p=0.87),
    transforms.RandomGrayscale(p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
