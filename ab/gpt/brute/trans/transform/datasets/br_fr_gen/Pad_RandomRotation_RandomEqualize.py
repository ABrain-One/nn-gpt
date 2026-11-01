import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(120, 31, 162), padding_mode='symmetric'),
    transforms.RandomRotation(degrees=28),
    transforms.RandomEqualize(p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
