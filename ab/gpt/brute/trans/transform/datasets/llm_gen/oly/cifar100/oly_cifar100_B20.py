import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp;&nbsp;return transforms.Compose([<br>
&nbsp;&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.5),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.0)], p=0.5),<br>
&nbsp;&nbsp;&nbsp;transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),<br>
&nbsp;&nbsp;&nbsp;transforms.Resize(64),<br>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br>
&nbsp;&nbsp;&nbsp;transforms.Normalize(*norm)<br>
&nbsp;&nbsp;])<br>