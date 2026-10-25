import torch<br>
from torchvision import transforms<br>
<br>
def transform(norm):<br>
&nbsp;&nbsp;return transforms.Compose([<br>
&nbsp;&nbsp;&nbsp;transforms.RandomResizedCrop(32, scale=(0.6, 0.9), ratio=(0.9, 1.2)),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.5),<br>
&nbsp;&nbsp;&nbsp;transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br>
&nbsp;&nbsp;&nbsp;transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.4)),<br>
&nbsp;&nbsp;&nbsp;transforms.Resize((64, 64)),<br>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br>
&nbsp;&nbsp;&nbsp;transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])<br>
&nbsp;&nbsp;])<br>