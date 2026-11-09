import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(115, 34, 60), padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomSolarize(threshold=150, p=0.27),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
