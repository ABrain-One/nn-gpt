```python
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomVerticalFlip(p=0.65),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.9),
        transforms.Pad(padding=2, fill=(110, 135, 40), padding_mode='symmetric'),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, p=0.7),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
```