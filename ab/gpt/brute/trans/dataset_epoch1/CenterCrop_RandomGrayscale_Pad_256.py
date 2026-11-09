import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomGrayscale(p=0.23),
    transforms.Pad(padding=5, fill=(141, 8, 50), padding_mode='symmetric'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
