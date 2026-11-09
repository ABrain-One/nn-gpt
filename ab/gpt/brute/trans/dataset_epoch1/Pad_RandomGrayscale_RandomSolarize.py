import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(12, 21, 49), padding_mode='reflect'),
    transforms.RandomGrayscale(p=0.67),
    transforms.RandomSolarize(threshold=148, p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
