import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(192, 217, 149), padding_mode='edge'),
    transforms.ColorJitter(brightness=0.86, contrast=1.17, saturation=1.13, hue=0.06),
    transforms.RandomPosterize(bits=7, p=0.49),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
