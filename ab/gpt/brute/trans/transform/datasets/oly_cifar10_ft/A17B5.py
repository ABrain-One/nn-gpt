import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(238, 174, 105), padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(p=0.58),
    transforms.RandomRotation(degrees=13),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])