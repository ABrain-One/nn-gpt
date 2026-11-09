import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.41),
    transforms.RandomEqualize(p=0.51),
    transforms.RandomHorizontalFlip(p=0.74),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
