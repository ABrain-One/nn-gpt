import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.64),
    transforms.RandomInvert(p=0.2),
    transforms.Pad(padding=5, fill=(19, 53, 106), padding_mode='symmetric'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
