import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(98, 97, 72), padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(p=0.33),
    transforms.RandomEqualize(p=0.23),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
