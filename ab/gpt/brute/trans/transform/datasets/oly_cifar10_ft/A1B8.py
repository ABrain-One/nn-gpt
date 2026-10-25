<td>
<pre>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(37, 35, 141), padding_mode='reflect'),
    transforms.RandomAutocontrast(p=0.85),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
</pre>
</td>