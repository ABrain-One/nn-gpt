import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(248, 98, 250), padding_mode='symmetric'),
    transforms.RandomPosterize(bits=7, p=0.62),
    transforms.RandomInvert(p=0.23),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
