import torch<br/>
from torchvision import transforms<br/>
from PIL import Image<br/>
<br/>
def transform(norm):<br/>
&nbsp;&nbsp;return transforms.Compose([<br/>
&nbsp;&nbsp;&nbsp;transforms.Resize(96),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.5),<br/>
&nbsp;&nbsp;&nbsp;transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomApply([\n&nbsp;&nbsp;transforms.RandomAffine(\n&nbsp;&nbsp;&nbsp;degrees=15,\n&nbsp;&nbsp;&nbsp;translate=(0.15, 0.15),\n&nbsp;&nbsp;&nbsp;scale=(0.9, 1.2),\n&nbsp;&nbsp;&nbsp;shear=(-5.0, 5.0),\n&nbsp;&nbsp;&nbsp;resample=Image.BICUBIC,\n&nbsp;&nbsp;&nbsp;fillcolor=(0.4914, 0.4822, 0.4465)\n\t)&nbsp;],&nbsp;p=0.8),<br/>
&nbsp;&nbsp;&nbsp;transforms.GaussianBlur(kernel_size=5,&sigma;=(0.5,&nbsp;2.0)),<br/>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br/>
&nbsp;&nbsp;&nbsp;transforms.Normalize(**norm)<br/>
&nbsp;&nbsp;])<br/>