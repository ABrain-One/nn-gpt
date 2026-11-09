import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.57),
    transforms.RandomGrayscale(p=0.56),
    transforms.RandomPosterize(bits=6, p=0.24),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
