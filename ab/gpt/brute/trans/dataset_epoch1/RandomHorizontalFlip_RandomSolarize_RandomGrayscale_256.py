import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.66),
    transforms.RandomSolarize(threshold=75, p=0.48),
    transforms.RandomGrayscale(p=0.75),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
