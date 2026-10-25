import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(26, 53, 115), padding_mode='edge'),
    transforms.RandomPosterize(bits=5, p=0.89),
    transforms.RandomEqualize(p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
