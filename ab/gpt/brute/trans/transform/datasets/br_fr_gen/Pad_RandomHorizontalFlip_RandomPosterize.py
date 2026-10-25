import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(234, 202, 155), padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.53),
    transforms.RandomPosterize(bits=4, p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
