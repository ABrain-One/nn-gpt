import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(234, 202, 35), padding_mode='symmetric'),
    transforms.RandomPosterize(bits=8, p=0.59),
    transforms.RandomHorizontalFlip(p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])