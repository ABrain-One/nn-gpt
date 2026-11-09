import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.Pad(padding=4, fill=(99, 159, 88), padding_mode='reflect'),
    transforms.RandomGrayscale(p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
