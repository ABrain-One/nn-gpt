import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(226, 148, 191), padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.72),
    transforms.RandomGrayscale(p=0.44),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
