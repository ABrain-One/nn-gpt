import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.15),
    transforms.Pad(padding=5, fill=(3, 88, 97), padding_mode='symmetric'),
    transforms.RandomSolarize(threshold=201, p=0.24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
