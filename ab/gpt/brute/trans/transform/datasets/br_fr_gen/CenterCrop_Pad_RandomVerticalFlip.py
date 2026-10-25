import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.Pad(padding=2, fill=(62, 186, 195), padding_mode='symmetric'),
    transforms.RandomVerticalFlip(p=0.47),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
