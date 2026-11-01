import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(3, 223, 282), padding_mode='edge'),
    transforms.RandomGrayscale(p=0.8),
    transforms.RandomAutocontrast(p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)]
)