import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.Pad(padding=0, fill=(118, 243, 208), padding_mode='constant'),
    transforms.RandomSolarize(threshold=197, p=0.17),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
