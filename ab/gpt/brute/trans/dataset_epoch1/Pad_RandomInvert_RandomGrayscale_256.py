import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(214, 0, 134), padding_mode='reflect'),
    transforms.RandomInvert(p=0.21),
    transforms.RandomGrayscale(p=0.69),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
