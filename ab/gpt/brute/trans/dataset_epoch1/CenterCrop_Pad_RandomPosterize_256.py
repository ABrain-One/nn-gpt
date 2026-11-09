import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.Pad(padding=3, fill=(28, 250, 207), padding_mode='constant'),
    transforms.RandomPosterize(bits=5, p=0.44),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
