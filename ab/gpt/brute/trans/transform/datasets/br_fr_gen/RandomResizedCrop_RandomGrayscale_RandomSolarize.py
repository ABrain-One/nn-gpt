import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.89), ratio=(1.21, 2.04)),
    transforms.RandomGrayscale(p=0.26),
    transforms.RandomSolarize(threshold=194, p=0.52),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
