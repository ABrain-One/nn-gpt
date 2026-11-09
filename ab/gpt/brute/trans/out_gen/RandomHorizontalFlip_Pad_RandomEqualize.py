import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.16),
    transforms.Pad(padding=3, fill=(134, 60, 215), padding_mode='symmetric'),
    transforms.RandomEqualize(p=0.52),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
