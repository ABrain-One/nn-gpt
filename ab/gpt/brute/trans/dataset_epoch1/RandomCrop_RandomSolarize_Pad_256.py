import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.RandomSolarize(threshold=58, p=0.12),
    transforms.Pad(padding=1, fill=(38, 215, 144), padding_mode='edge'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
