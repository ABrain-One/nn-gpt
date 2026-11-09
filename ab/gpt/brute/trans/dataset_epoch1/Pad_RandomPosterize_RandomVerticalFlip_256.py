import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(206, 1, 6), padding_mode='symmetric'),
    transforms.RandomPosterize(bits=7, p=0.39),
    transforms.RandomVerticalFlip(p=0.63),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
