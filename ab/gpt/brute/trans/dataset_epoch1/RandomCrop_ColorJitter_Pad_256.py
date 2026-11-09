import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.ColorJitter(brightness=0.83, contrast=0.96, saturation=1.13, hue=0.01),
    transforms.Pad(padding=4, fill=(90, 17, 243), padding_mode='symmetric'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
