import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(29, 191, 185), padding_mode='symmetric'),
    transforms.CenterCrop(size=31),
    transforms.RandomInvert(p=0.47),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
