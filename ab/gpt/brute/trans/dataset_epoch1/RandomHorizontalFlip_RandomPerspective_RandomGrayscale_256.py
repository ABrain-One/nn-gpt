import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.24),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.43),
    transforms.RandomGrayscale(p=0.23),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
