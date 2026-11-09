import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(49, 194, 240), padding_mode='reflect'),
    transforms.RandomGrayscale(p=0.58),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.28),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
