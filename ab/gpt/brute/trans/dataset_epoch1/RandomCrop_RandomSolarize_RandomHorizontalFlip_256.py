import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.RandomSolarize(threshold=98, p=0.4),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
