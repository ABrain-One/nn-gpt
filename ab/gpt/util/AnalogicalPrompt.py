NN_ALTER_COMPACT_V3_PROMPT = """Output only valid Python code in XML tags.
Start with <nn> and end with </nn>.
No prose, no markdown, no comments, no <think>.

Required structure (must all exist):
- class Net(nn.Module)
- method __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device)
- method forward(self, x)
- method train_setup(self, prm)
- method learn(self, train_data)

Hard constraints:
1. Keep class name `Net`.
2. Use only standard PyTorch.
3. Do not use torchvision model classes.
4. Return full runnable code.

Analogical exemplar (successful runnable pattern):
```python
import torch
import torch.nn as nn

def supported_hyperparameters():
    return {{'lr', 'momentum', 'dropout'}}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        c_in = in_shape[1]
        n_cls = out_shape[0]
        p = prm['dropout']
        self.features = nn.Sequential(
            nn.Conv2d(c_in, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=p),
            nn.Linear(256, n_cls)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
```
Use the same reliability patterns as this exemplar:
- channel index from in_shape[1]
- explicit supported_hyperparameters()
- stable train_setup/learn with SGD + CrossEntropyLoss
- forward returns logits tensor directly

Self-check before final output:
- Ensure class Net exists.
- Ensure all five required methods exist.
- Ensure output is enclosed by <nn>...</nn>.
- If any check fails, regenerate internally and then output only final valid code.

Main model (accuracy {accuracy}):
```python
{nn_code}
```

Reference target accuracy from support model: {addon_accuracy}
Now output only <nn>...</nn>.
"""

NN_ALTER_ANALOGICAL_PREFILL = "<nn>\nimport torch\nimport torch.nn as nn\n\n"


def get_nn_alter_analogical_prompt() -> str:
    return NN_ALTER_COMPACT_V3_PROMPT


def get_nn_alter_analogical_prefill() -> str:
    return NN_ALTER_ANALOGICAL_PREFILL
