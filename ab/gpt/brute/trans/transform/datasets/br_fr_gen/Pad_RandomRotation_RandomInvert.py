import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(199, 77, 171), padding_mode='symmetric'),
    transforms.RandomRotation(degrees=17),
    transforms.RandomInvert(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
