import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(223, 7, 161), padding_mode='reflect'),
    transforms.RandomSolarize(threshold=198, p=0.35),
    transforms.RandomGrayscale(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])