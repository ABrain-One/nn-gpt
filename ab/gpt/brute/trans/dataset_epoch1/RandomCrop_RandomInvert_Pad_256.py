import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomInvert(p=0.33),
    transforms.Pad(padding=1, fill=(126, 184, 43), padding_mode='symmetric'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
