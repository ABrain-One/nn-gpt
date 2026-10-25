```python
import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([
        T.Pad(padding=3, fill=(103, 36, 31), padding_mode='constant'),
        T.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.8, 1.2))], p=0.5),
        T.RandomInvert(p=0.2),
        T.RandomSolarize(195, p=0.3),
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(**norm)
    ])
```