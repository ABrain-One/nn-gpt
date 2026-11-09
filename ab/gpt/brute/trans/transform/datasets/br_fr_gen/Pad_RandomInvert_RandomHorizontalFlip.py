import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(115, 217, 221), padding_mode='symmetric'),
    transforms.RandomInvert(p=0.31),
    transforms.RandomHorizontalFlip(p=0.77),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
