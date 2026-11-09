import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.58),
    transforms.Pad(padding=4, fill=(207, 5, 10), padding_mode='symmetric'),
    transforms.RandomRotation(degrees=19),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
