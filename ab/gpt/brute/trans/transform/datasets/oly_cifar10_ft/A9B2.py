import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(219, 105, 67), padding_mode='reflect'),
    transforms.RandomPosterize(bits=4, p=0.49),
    transforms.RandomGrayscale(p=0.38),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])