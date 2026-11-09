import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(129, 215, 229), padding_mode='symmetric'),
    transforms.RandomVerticalFlip(p=0.31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
