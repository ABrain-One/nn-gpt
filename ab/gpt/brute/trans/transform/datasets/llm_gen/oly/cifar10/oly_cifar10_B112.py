import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp;&nbsp;return transforms.Compose([<br>
&nbsp;&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.2),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomVerticalFlip(p=0.2),<br>
&nbsp;&nbsp;&nbsp;transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br>
&nbsp;&nbsp;&nbsp;transforms.GaussianBlur(kernel_size=3, sigma=1.5, p=0.5),<br>
&nbsp;&nbsp;&nbsp;transforms.Pad(padding=5, fill=(125, 122, 112), padding_mode='constant'),<br>
&nbsp;&nbsp;&nbsp;transforms.Resize((64, 64)),<br>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br>
&nbsp;&nbsp;&nbsp;transforms.Normalize(*norm)<br>
&nbsp;&nbsp;])<br>