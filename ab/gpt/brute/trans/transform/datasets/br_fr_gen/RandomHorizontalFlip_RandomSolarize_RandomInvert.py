import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.63),
    transforms.RandomSolarize(threshold=33, p=0.28),
    transforms.RandomInvert(p=0.77),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
