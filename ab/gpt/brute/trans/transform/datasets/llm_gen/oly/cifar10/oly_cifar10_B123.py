<td>
<p><code>
import torch<br>
from torchvision import transforms<br>
</code></p>
<div style="margin-left:2em;"><code>
def transform(norm):<br>
&nbsp;&nbsp;return transforms.Compose([<br>
&nbsp;&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.5),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomVerticalFlip(p=0.1),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomAffine(<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;degrees=(-5, 5),<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;translate=(0.125, 0.125),<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;shear=(-2, 2),<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fill=(158, 163, 1)<br>
&nbsp;&nbsp;&nbsp;&nbsp;),<br>
&nbsp;&nbsp;&nbsp;transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.3),<br>
&nbsp;&nbsp;&nbsp;transforms.Resize(224),<br>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br>
&nbsp;&nbsp;&nbsp;transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),<br>
&nbsp;&nbsp;]<br>
</code></div>
</div>
</td>