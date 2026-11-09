import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(64, 31, 105), padding_mode='constant'),
    transforms.CenterCrop(size=25),
    transforms.RandomGrayscale(p=0.6),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
