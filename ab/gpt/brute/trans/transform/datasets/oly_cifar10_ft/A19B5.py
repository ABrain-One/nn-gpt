import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(160, 13, 144), padding_mode='symmetric'),
    transforms.RandomEqualize(p=0.61),
    transforms.RandomGrayscale(p=0.64),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])