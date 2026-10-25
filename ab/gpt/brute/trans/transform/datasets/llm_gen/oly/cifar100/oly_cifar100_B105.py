import torch<br>
from torchvision import transforms<br>
<br>
def transform():<br>
&nbsp;&nbsp;return transforms.Compose([<br>
&nbsp;&nbsp;&nbsp;transforms.RandomVerticalFlip(p=0.2),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.2),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomAffine(<br>
&nbsp;&nbsp;&nbsp;&nbsp;degrees=5,<br>
&nbsp;&nbsp;&nbsp;&nbsp;translate=(0.1, 0.1),<br>
&nbsp;&nbsp;&nbsp;&nbsp;scale=(0.95, 1.05),<br>
&nbsp;&nbsp;&nbsp;&nbsp;shear=(-2, 2)<br>
&nbsp;&nbsp;&nbsp;),<br>
&nbsp;&nbsp;&nbsp;transforms.ColorJitter(<br>
&nbsp;&nbsp;&nbsp;&nbsp;brightness=0.2,<br>
&nbsp;&nbsp;&nbsp;&nbsp;contrast=0.2,<br>
&nbsp;&nbsp;&nbsp;&nbsp;saturation=0.2,<br>
&nbsp;&nbsp;&nbsp;&nbsp;hue=0.1<br>
&nbsp;&nbsp;&nbsp;),<br>
&nbsp;&nbsp;&nbsp;transforms.Resize((224, 224)),<br>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br>
&nbsp;&nbsp;&nbsp;transforms.Normalize(<br>
&nbsp;&nbsp;&nbsp;&nbsp;mean=[0.4914, 0.4822, 0.4465],<br>
&nbsp;&nbsp;&nbsp;&nbsp;std=[0.2023, 0.1994, 0.2010]<br>
&nbsp;&nbsp;&nbsp;)<br>
&nbsp;&nbsp;])<br>