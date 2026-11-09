import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(3, 247, 127), padding_mode='edge'),
    transforms.RandomGrayscale(p=0.89),
    transforms.RandomAdjustSharpness(sharpness_factor=1.46, p=0.49),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
