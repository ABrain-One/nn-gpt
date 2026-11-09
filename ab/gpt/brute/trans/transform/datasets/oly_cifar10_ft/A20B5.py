<name>Transform1</name>
    <src>import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(131, 108, 239), padding_mode='edge'),
    transforms.RandomPosterize(bits=6, p=0.74),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
</src>