import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class ComplexExpert(nn.Module):

    def __init__(self, feature_dim, hidden_dim, output_dim):
        super(ComplexExpert, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if not torch.is_complex(x):
            x = x.type(torch.complex64)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.size(-1) != self.feature_dim:
            if x.size(-1) > self.feature_dim:
                x = x[:, :self.feature_dim]
            else:
                padding = self.feature_dim - x.size(-1)
                x = nn.functional.pad(x, (0, padding))
        x = nn.functional.relu(self.fc1(x))
        x_real = self.dropout(x.real)
        x_imag = self.dropout(x.imag)
        x = x_real + 1j * x_imag
        x = nn.functional.relu(self.fc2(x))
        x_real = self.dropout(x.real)
        x_imag = self.dropout(x.imag)
        x = x_real + 1j * x_imag
        x = self.fc3(x)
        return x

class ComplexGate(nn.Module):

    def __init__(self, feature_dim, n_experts, hidden_dim=32):
        super(ComplexGate, self).__init__()
        self.feature_dim = feature_dim
        self.n_experts = n_experts
        self.top_k = 2
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if torch.is_complex(x):
            x = x.abs()
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.size(-1) != self.feature_dim:
            if x.size(-1) > self.feature_dim:
                x = x[:, :self.feature_dim]
            else:
                padding = self.feature_dim - x.size(-1)
                x = nn.functional.pad(x, (0, padding))
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        gate_logits = self.fc2(x)
        if self.training:
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
        (top_k_logits, top_k_indices) = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = nn.functional.softmax(top_k_logits, dim=-1)
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        return (gates, top_k_indices)

class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 8
        self.top_k = 2
        self.in_channels = in_shape[1] if len(in_shape) > 1 else 1
        self.in_height = in_shape[2] if len(in_shape) > 2 else 28
        self.in_width = in_shape[3] if len(in_shape) > 3 else 28
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        self.conv1 = nn.Conv2d(self.in_channels, 10, kernel_size=5, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)
        with torch.no_grad():
            tmp_input = torch.zeros(1, self.in_channels, self.in_height, self.in_width).type(torch.complex64)
            tmp_features = self._extract_features(tmp_input)
            self.feature_dim = tmp_features.view(-1).size(0)
        self.hidden_dim = min(128, max(32, self.feature_dim // 16))
        self.experts = nn.ModuleList([ComplexExpert(self.feature_dim, self.hidden_dim, self.output_dim) for _ in range(self.n_experts)])
        self.gate = ComplexGate(self.feature_dim, self.n_experts, self.hidden_dim // 2)
        self.to(device)

    def _extract_features(self, x):
        x = x.view(-1, self.in_channels, self.in_height, self.in_width)
        x = nn.functional.relu(self.bn(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        return x

    def forward(self, x):
        try:
            if not torch.is_complex(x):
                x = x.type(torch.complex64)
            batch_size = x.size(0)
            features = self._extract_features(x)
            features_flat = features.view(batch_size, -1)
            (gate_weights, top_k_indices) = self.gate(features_flat)
            output = torch.zeros(batch_size, self.output_dim, device=self.device, dtype=torch.complex64)
            for i in range(self.n_experts):
                expert_mask = (top_k_indices == i).any(dim=1)
                if expert_mask.any():
                    expert_output = self.experts[i](features_flat)
                    weighted_output = expert_output * gate_weights[:, i].unsqueeze(-1).type(torch.complex64)
                    output += weighted_output
            output = output.abs()
            output = nn.functional.log_softmax(output, dim=1)
            return output
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print('GPU out of memory! Clearing cache...')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return torch.zeros(x.size(0), self.output_dim, device=self.device)
            else:
                raise e

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm.get('lr', 0.01), momentum=prm.get('momentum', 0.9))

    def learn(self, train_data):
        self.train()
        total_loss = 0
        num_batches = 0
        for (batch_idx, (inputs, labels)) in enumerate(train_data):
            try:
                if batch_idx % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)
                if inputs.size(0) > 32:
                    inputs = inputs[:32]
                    labels = labels[:32]
                self.optimizer.zero_grad()
                outputs = self(inputs)
                if outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
                if labels.dim() > 1:
                    labels = labels.view(-1)
                loss = self.criteria(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=3)
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                del inputs, labels, outputs, loss
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f'OOM at batch {batch_idx}, skipping...')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    print(f'Training error: {e}')
                    continue
        return total_loss / max(num_batches, 1)