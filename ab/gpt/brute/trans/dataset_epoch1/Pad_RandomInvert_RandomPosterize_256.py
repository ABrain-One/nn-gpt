import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(110, 222, 5), padding_mode='constant'),
    transforms.RandomInvert(p=0.67),
    transforms.RandomPosterize(bits=8, p=0.47),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
