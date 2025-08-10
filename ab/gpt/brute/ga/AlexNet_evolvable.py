# AlexNet_evolvable.py
import torch
import torch.nn as nn
import random

SEARCH_SPACE = {
    'conv1_filters': [32, 64, 96],
    'conv1_kernel': [7, 9, 11],
    'conv1_stride': [3, 4],
    'conv2_filters': [128, 192, 256],
    'conv2_kernel': [3, 5],
    'conv3_filters': [256, 384, 440],
    'conv4_filters': [256, 384],
    'conv5_filters': [192, 256],
    'fc1_neurons': [2048, 3072, 4096],
    'fc2_neurons': [2048, 3072, 4096],
    'lr': [0.001, 0.005, 0.01],
    'momentum': [0.85, 0.9, 0.95],
    'dropout': [0.4, 0.5, 0.6],
}

def create_random_chromosome():
    chromosome = {}
    for key, values in SEARCH_SPACE.items():
        chromosome[key] = random.choice(values)
    return chromosome

def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=prm['lr'],
            momentum=prm['momentum']
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, chromosome: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.chromosome = chromosome

        layers = []
        in_channels = in_shape[0]

        layers += [
            nn.Conv2d(in_channels, chromosome['conv1_filters'], kernel_size=chromosome['conv1_kernel'],
                      stride=chromosome['conv1_stride'], padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = chromosome['conv1_filters']

        layers += [
            nn.Conv2d(in_channels, chromosome['conv2_filters'], kernel_size=chromosome['conv2_kernel'], padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = chromosome['conv2_filters']

        layers += [
            nn.Conv2d(in_channels, chromosome['conv3_filters'], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        in_channels = chromosome['conv3_filters']

        layers += [
            nn.Conv2d(in_channels, chromosome['conv4_filters'], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        in_channels = chromosome['conv4_filters']

        layers += [
            nn.Conv2d(in_channels, chromosome['conv5_filters'], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        in_channels = chromosome['conv5_filters']

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        dropout_p = chromosome['dropout']
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(classifier_input_features, chromosome['fc1_neurons']),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(chromosome['fc1_neurons'], chromosome['fc2_neurons']),
            nn.ReLU(inplace=True),
            nn.Linear(chromosome['fc2_neurons'], out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def generate_model_code_string(chromosome: dict, in_shape: tuple = (3, 32, 32), out_shape: tuple = (10,)) -> str:
    code_lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        "def supported_hyperparameters():",
        "    return {'lr', 'momentum', 'dropout'}",
        "",
        "",
        "class Net(nn.Module):",
        "    def train_setup(self, prm):",
        "        self.to(self.device)",
        "        self.criteria = (nn.CrossEntropyLoss().to(self.device),)",
        "        self.optimizer = torch.optim.SGD(",
        "            self.parameters(),",
        f"            lr=prm['lr'],",
        f"            momentum=prm['momentum']",
        "        )",
        "",
        "    def learn(self, train_data):",
        "        self.train()",
        "        for inputs, labels in train_data:",
        "            inputs, labels = inputs.to(self.device), labels.to(self.device)",
        "            self.optimizer.zero_grad()",
        "            outputs = self(inputs)",
        "            loss = self.criteria[0](outputs, labels)",
        "            loss.backward()",
        "            torch.nn.utils.clip_grad_norm_(self.parameters(), 3)",
        "            self.optimizer.step()",
        "",
        "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:",
        f"        super().__init__()",
        f"        self.device = device",
        # --- Feature Extractor ---
        "        layers = []",
        f"        in_channels = in_shape[0]",
        "",
        # Layer 1
        f"        layers += [",
        f"            nn.Conv2d(in_channels, {chromosome['conv1_filters']}, kernel_size={chromosome['conv1_kernel']},",
        f"                      stride={chromosome['conv1_stride']}, padding=2),",
        f"            nn.ReLU(inplace=True),",
        f"            nn.MaxPool2d(kernel_size=2, stride=2),",
        f"        ]",
        f"        in_channels = {chromosome['conv1_filters']}",
        "",
        # Layer 2
        f"        layers += [",
        f"            nn.Conv2d(in_channels, {chromosome['conv2_filters']}, kernel_size={chromosome['conv2_kernel']}, padding=2),",
        f"            nn.ReLU(inplace=True),",
        f"            nn.MaxPool2d(kernel_size=2, stride=2),",
        f"        ]",
        f"        in_channels = {chromosome['conv2_filters']}",
        "",
        # Layer 3
        f"        layers += [",
        f"            nn.Conv2d(in_channels, {chromosome['conv3_filters']}, kernel_size=3, padding=1),",
        f"            nn.ReLU(inplace=True),",
        f"        ]",
        f"        in_channels = {chromosome['conv3_filters']}",
        "",
        # Layer 4
        f"        layers += [",
        f"            nn.Conv2d(in_channels, {chromosome['conv4_filters']}, kernel_size=3, padding=1),",
        f"            nn.ReLU(inplace=True),",
        f"        ]",
        f"        in_channels = {chromosome['conv4_filters']}",
        "",
        # Layer 5
        f"        layers += [",
        f"            nn.Conv2d(in_channels, {chromosome['conv5_filters']}, kernel_size=3, padding=1),",
        f"            nn.ReLU(inplace=True),",
        f"            nn.MaxPool2d(kernel_size=2, stride=2),",
        f"        ]",
        f"        in_channels = {chromosome['conv5_filters']}",
        "        self.features = nn.Sequential(*layers)",
        "        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))",
        "",
        # --- Classifier ---
        f"        dropout_p = prm['dropout']",
        f"        classifier_input_features = in_channels * 6 * 6",
        "        self.classifier = nn.Sequential(",
        "            nn.Dropout(p=dropout_p),",
        f"            nn.Linear(classifier_input_features, {chromosome['fc1_neurons']}),",
        "            nn.ReLU(inplace=True),",
        "            nn.Dropout(p=dropout_p),",
        f"            nn.Linear({chromosome['fc1_neurons']}, {chromosome['fc2_neurons']}),",
        "            nn.ReLU(inplace=True),",
        f"            nn.Linear({chromosome['fc2_neurons']}, out_shape[0]),",
        "        )",
        "",
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
        "        x = self.features(x)",
        "        x = self.avgpool(x)",
        "        x = torch.flatten(x, 1)",
        "        x = self.classifier(x)",
        "        return x",
        "",
        "# --- Chromosome used to generate this model ---",
        f"# Chromosome: {chromosome}",
        ""
    ]
    return "\n".join(code_lines)
