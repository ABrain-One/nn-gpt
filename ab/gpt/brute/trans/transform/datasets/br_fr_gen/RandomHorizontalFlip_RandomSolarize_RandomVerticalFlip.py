import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.84),
    transforms.RandomSolarize(threshold=90, p=0.46),
    transforms.RandomVerticalFlip(p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
