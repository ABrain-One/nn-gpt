import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomSolarize(threshold=22, p=0.62),
    transforms.RandomGrayscale(p=0.8),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
