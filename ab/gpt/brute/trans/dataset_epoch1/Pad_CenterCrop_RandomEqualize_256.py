import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(118, 45, 2), padding_mode='symmetric'),
    transforms.CenterCrop(size=32),
    transforms.RandomEqualize(p=0.89),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
