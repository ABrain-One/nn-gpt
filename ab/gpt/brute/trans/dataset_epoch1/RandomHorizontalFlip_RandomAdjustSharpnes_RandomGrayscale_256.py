import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.38),
    transforms.RandomAdjustSharpness(sharpness_factor=1.43, p=0.54),
    transforms.RandomGrayscale(p=0.69),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
