import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(22, 24, 39), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.49, p=0.43),
    transforms.RandomGrayscale(p=0.59),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
