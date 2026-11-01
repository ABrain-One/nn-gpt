import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(68, 164, 122), padding_mode='symmetric'),
    transforms.RandomCrop(size=26),
    transforms.RandomGrayscale(p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
