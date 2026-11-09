import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(160, 104, 22), padding_mode='symmetric'),
    transforms.RandomEqualize(p=0.87),
    transforms.RandomAutocontrast(p=0.17),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
