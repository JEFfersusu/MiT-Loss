# class MiT_Loss(nn.Module):
#     def __init__(self, T=0.5, lambda_entropy=0.1):
#         super().__init__()
#         self.ce = nn.CrossEntropyLoss()
#         self.T = T
#         self.lambda_entropy = lambda_entropy

#     def forward(self, outputs, targets):
#         scaled_outputs = outputs / self.T
#         ce_loss = self.ce(scaled_outputs, targets)
#         probs = torch.softmax(scaled_outputs, dim=1)
#         entropy = - (probs * torch.log(probs + 1e-10)).sum(dim=1).mean()

#         return ce_loss - self.lambda_entropy * entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

def _softplus_floor(x, eps=1e-6):
    return F.softplus(x) + eps

def _collect_logits_labels(model: nn.Module, data_loader, device):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits_list.append(outputs)
            labels_list.append(labels)
    return torch.cat(logits_list), torch.cat(labels_list)

class MiTLoss_WithTrainCalibration(nn.Module):
    def __init__(self, num_classes: int, train_loader, model: nn.Module, device):
        super().__init__()
        self.num_classes = int(num_classes)
        self.device = device
        self.model = model
        self.train_loader = train_loader

        # ---- Initialize temperature using training set ----
        init_tau = self._initialize_temperature()
        self.tau = nn.Parameter(init_tau)

        # ---- Running class histogram (for empirical label entropy H*) ----
        self.register_buffer("class_counts", torch.ones(self.num_classes))
        self.register_buffer("total_seen", torch.tensor(self.num_classes, dtype=torch.long))

        # ---- Dual-averaged λ ----
        self.register_buffer("lambda_entropy", torch.tensor(0.1))
        self.register_buffer("dual_updates", torch.tensor(0.1, dtype=torch.long))

        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def _initialize_temperature(self):
        warnings.warn("Initializing temperature using the training set.")
        logits, labels = _collect_logits_labels(self.model, self.train_loader, self.device)
        logits, labels = logits.to(self.device), labels.to(self.device)

        logT = torch.tensor(0.0, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            T = torch.exp(logT)
            log_probs = F.log_softmax(logits / T, dim=1)
            nll = F.nll_loss(log_probs, labels, reduction='mean')
            nll.backward()
            return nll

        opt.step(closure)
        T_star = torch.exp(logT).detach()               
        tau0 = torch.log(torch.expm1(T_star).clamp_min(1e-12))
        return tau0

    @torch.no_grad()
    def _update_label_entropy(self, targets: torch.Tensor):
        dev = self.class_counts.device
        targets = targets.to(dev, dtype=torch.long)
        binc = torch.bincount(targets, minlength=self.num_classes).to(dev, dtype=self.class_counts.dtype)
        self.class_counts += binc
        self.total_seen += targets.numel()

    @torch.no_grad()
    def _empirical_label_entropy(self) -> torch.Tensor:
        probs = self.class_counts / self.class_counts.sum()
        logp = torch.log(probs.clamp_min(1e-12))
        H_star = -(probs * logp).sum()
        return H_star

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        dev = logits.device
        targets = targets.to(dev, dtype=torch.long)

        # 1) Update H* from data
        self._update_label_entropy(targets)
        H_star = self._empirical_label_entropy()
        H_max = math.log(self.num_classes + 1e-12)

        # 2) Temperature scaling
        T = _softplus_floor(self.tau).clamp(1e-3, 500.0)
        scaled = logits / T

        # 3) Cross-entropy loss
        ce_loss = self.ce(scaled, targets)

        # 4) Entropy of predictive distribution
        log_probs = F.log_softmax(scaled, dim=1)
        probs = log_probs.exp()
        H = -(probs * log_probs).sum(dim=1).mean()

        # 5) Update λ via dual-averaging
        with torch.no_grad():
            d = (H_star - H) / max(H_max, 1e-12)
            d = torch.clamp(d, min=0.0)
            self.dual_updates += 1
            new_lambda = (self.lambda_entropy * (self.dual_updates - 1) + d) / self.dual_updates
            new_lambda = torch.clamp(new_lambda, 0.0, 0.5)
            self.lambda_entropy.copy_(new_lambda)

        # 6) Combined loss
        loss = ce_loss - self.lambda_entropy * H

        stats = {
            "loss": loss.detach(),
            "ce": ce_loss.detach(),
            "H": H.detach(),
            "H_star": H_star.detach(),
            "lambda": self.lambda_entropy.detach(),
            "T": T.detach()
        }
        return loss, stats
