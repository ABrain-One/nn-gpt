import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.44),
    transforms.RandomGrayscale(p=0.86),
    transforms.RandomSolarize(threshold=168, p=0.12),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
