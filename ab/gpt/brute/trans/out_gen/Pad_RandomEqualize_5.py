import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(109, 241, 162), padding_mode='reflect'),
    transforms.RandomEqualize(p=0.79),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
