class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.n_experts = 8
        self.top_k = 2
        self.feature_extractor = ImprovedFeatureExtractor(in_channels=in_shape[1])
        self.feature_dim = self.feature_extractor.output_dim
        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape
        self.hidden_dim = 256
        self.experts = nn.ModuleList([ExpertWithSpecialization(self.feature_dim, self.hidden_dim, self.output_dim, i) for i in range(self.n_experts)])
        self.gate = ImprovedGate(self.feature_dim, self.n_experts, 64)
        self.utilization_tracker = ExpertUtilizationTracker(self.n_experts, self.output_dim)
        self.load_balance_weight = 0.005
        self.diversity_weight = 0.002
        self.label_smoothing = 0.05
        self.to(device)
        self._print_memory_info()

    def _print_memory_info(self):
        param_count = sum((p.numel() for p in self.parameters()))
        param_size_mb = param_count * 4 / (1024 * 1024)
        print(f'Improved MoE-8 Model parameters: {param_count:,}')
        print(f'Model size: {param_size_mb:.2f} MB')
        print(f'Feature dim: {self.feature_dim}, Hidden dim: {self.hidden_dim}')

    def forward(self, x):
        features = self.feature_extractor(x)
        (gate_weights, top_k_indices, gate_logits) = self.gate(features)
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=2)
        gate_weights_expanded = gate_weights.unsqueeze(1)
        final_output = torch.sum(expert_outputs * gate_weights_expanded, dim=2)
        if self.training:
            self._last_gate_weights = gate_weights.detach()
            self._last_top_k_indices = top_k_indices.detach()
            active_outputs = []
            unique_experts = torch.unique(top_k_indices)
            for expert_idx in unique_experts[:4]:
                active_outputs.append(expert_outputs[:, :, expert_idx].mean(dim=0).detach())
            self._last_active_outputs = active_outputs
        return final_output

    def compute_load_balance_loss(self, gate_weights, expert_indices):
        if gate_weights is None or gate_weights.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        expert_usage = gate_weights.sum(dim=0)
        target_usage = gate_weights.sum() / self.n_experts
        balance_loss = F.mse_loss(expert_usage, target_usage.expand_as(expert_usage))
        return torch.clamp(balance_loss, 0.0, 1.0)

    def compute_diversity_loss(self, expert_outputs):
        if not expert_outputs or len(expert_outputs) < 2:
            return torch.tensor(0.0, device=self.device)
        similarities = []
        for i in range(min(len(expert_outputs), 3)):
            for j in range(i + 1, min(len(expert_outputs), 3)):
                sim = F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=0)
                similarities.append(sim)
        if similarities:
            return torch.stack(similarities).mean()
        return torch.tensor(0.0, device=self.device)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=prm.get('lr', 0.002), weight_decay=prm.get('weight_decay', 0.0005), betas=(0.9, 0.95), eps=1e-06)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-05)

    def learn(self, train_data):
        self.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        self.utilization_tracker.reset()
        for (inputs, labels) in train_data:
            inputs = inputs.to(self.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            main_loss = self.criteria(outputs, labels)
            load_loss = torch.tensor(0.0, device=self.device)
            div_loss = torch.tensor(0.0, device=self.device)
            if hasattr(self, '_last_gate_weights'):
                load_loss = self.compute_load_balance_loss(self._last_gate_weights, self._last_top_k_indices) * self.load_balance_weight
            if hasattr(self, '_last_active_outputs'):
                div_loss = self.compute_diversity_loss(self._last_active_outputs) * self.diversity_weight
            total_loss_batch = main_loss + load_loss + div_loss
            total_loss_batch.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()
            (_, predicted) = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if hasattr(self, '_last_gate_weights'):
                self.utilization_tracker.update(self._last_gate_weights, predicted, labels, self._last_top_k_indices)
            total_loss += total_loss_batch.item()
            num_batches += 1
            if hasattr(self, '_last_gate_weights'):
                delattr(self, '_last_gate_weights')
            if hasattr(self, '_last_top_k_indices'):
                delattr(self, '_last_top_k_indices')
            if hasattr(self, '_last_active_outputs'):
                delattr(self, '_last_active_outputs')
        self.scheduler.step()
        metrics = self.utilization_tracker.get_specialization_metrics()
        train_acc = 100.0 * correct / total
        print(f'Epoch Training Accuracy: {train_acc:.2f}%')
        print(f"Expert Usage Entropy: {metrics['usage_entropy']:.3f} (lower = more specialized)")
        print(f"Expert Usage Distribution: {metrics['usage_distribution']}")
        print(f"Expert Diversity (JS): {metrics['average_diversity']:.3f} (higher = more diverse)")
        print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        return total_loss / num_batches if num_batches > 0 else 0