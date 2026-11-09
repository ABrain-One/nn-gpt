import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(148, 75, 24), padding_mode='edge'),
    transforms.RandomGrayscale(p=0.24),
    transforms.RandomRotation(degrees=17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
