import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.12),
    transforms.Pad(padding=4, fill=(129, 84, 174), padding_mode='symmetric'),
    transforms.RandomPosterize(bits=7, p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
