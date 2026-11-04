import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(255, 234, 40), padding_mode='symmetric'),
    transforms.RandomCrop(size=27),
    transforms.RandomInvert(p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
