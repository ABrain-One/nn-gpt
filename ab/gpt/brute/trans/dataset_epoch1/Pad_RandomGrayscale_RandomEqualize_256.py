import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(22, 95, 60), padding_mode='reflect'),
    transforms.RandomGrayscale(p=0.64),
    transforms.RandomEqualize(p=0.61),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
