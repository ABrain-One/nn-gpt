import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(253, 92, 112), padding_mode='symmetric'),
    transforms.RandomVerticalFlip(p=0.32),
    transforms.RandomEqualize(p=0.85),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
