import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(10, 21, 19), padding_mode='symmetric'),
    transforms.RandomGrayscale(p=0.32),
    transforms.RandomInvert(p=0.35),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
