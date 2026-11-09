import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.89),
    transforms.RandomSolarize(threshold=13, p=0.45),
    transforms.Pad(padding=1, fill=(234, 162, 240), padding_mode='reflect'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
