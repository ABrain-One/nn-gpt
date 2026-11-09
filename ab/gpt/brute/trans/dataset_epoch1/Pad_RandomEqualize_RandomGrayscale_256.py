import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(8, 208, 151), padding_mode='symmetric'),
    transforms.RandomEqualize(p=0.21),
    transforms.RandomGrayscale(p=0.73),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
