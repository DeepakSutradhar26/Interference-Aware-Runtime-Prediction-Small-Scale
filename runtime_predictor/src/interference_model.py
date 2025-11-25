import torch
import torch.nn as nn
import numpy as np


class PitotModel(nn.Module):
    def __init__(
        self,
        n_workloads,
        n_platforms,
        workload_feat_dim,
        platform_feat_dim,
        emb_dim=32,
        s=6,
        hidden_w=[128, 128, 64],
        hidden_p=[128, 128, 64],
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.s = s

        # Workload tower
        layers_w = []
        w_in = workload_feat_dim
        for h in hidden_w:
            layers_w.append(nn.Linear(w_in, h))
            layers_w.append(nn.LayerNorm(h))
            layers_w.append(nn.ReLU())
            layers_w.append(nn.Dropout(0.1))
            w_in = h
        layers_w.append(nn.Linear(w_in, emb_dim))
        self.workload_net = nn.Sequential(*layers_w)

        # Platform tower
        layers_p = []
        p_in = platform_feat_dim
        for h in hidden_p:
            layers_p.append(nn.Linear(p_in, h))
            layers_p.append(nn.LayerNorm(h))
            layers_p.append(nn.ReLU())
            layers_p.append(nn.Dropout(0.1))
            p_in = h
        layers_p.append(nn.Linear(p_in, emb_dim))
        self.platform_net = nn.Sequential(*layers_p)

        # Interference
        self.v_s = nn.Parameter(torch.randn(s, emb_dim) * 0.1)
        self.v_g = nn.Parameter(torch.randn(s, emb_dim) * 0.1)

        self.act = nn.LeakyReLU(0.1)

    def forward(self, w_feats, p_feats):
        W = self.workload_net(w_feats)
        P = self.platform_net(p_feats)

        base = (W * P).sum(dim=1)

        inter = 0.0
        for t in range(self.s):
            ws = (W * self.v_s[t]).sum(dim=1)
            wg = (W * self.v_g[t]).sum(dim=1)
            inter += ws * self.act(wg)

        return base + inter


def train_pitot(model, dataset, optimizer, batch_size=128, epochs=40,
                device=torch.device("cpu"), verbose=True):

    model.to(device)
    train_idx = dataset["train_idx"]
    samples = dataset["samples"]
    workload = dataset["workload_opcodes"]
    platform = dataset["platform_feats"]
    y_log = dataset["y_log"]

    criterion = nn.HuberLoss(delta=1.0)

    n = len(train_idx)

    for ep in range(epochs):
        perm = np.random.permutation(train_idx)
        losses = []

        for start in range(0, n, batch_size):
            batch_idx = perm[start:start+batch_size]

            w_ids = [samples[i][0] for i in batch_idx]
            p_ids = [samples[i][1] for i in batch_idx]

            w = torch.tensor(workload[w_ids], dtype=torch.float32, device=device)
            p = torch.tensor(platform[p_ids], dtype=torch.float32, device=device)
            y_true = torch.tensor(y_log[batch_idx], dtype=torch.float32, device=device)

            y_pred = model(w, p)
            loss = criterion(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f"[Epoch {ep+1}/{epochs}] Loss={np.mean(losses):.4f}")

    return model


def predict_pitot(model, dataset, indices, device=torch.device("cpu")):
    model.eval()
    samples = dataset["samples"]
    workload = dataset["workload_opcodes"]
    platform = dataset["platform_feats"]

    with torch.no_grad():
        w_ids = [samples[i][0] for i in indices]
        p_ids = [samples[i][1] for i in indices]

        w = torch.tensor(workload[w_ids], dtype=torch.float32, device=device)
        p = torch.tensor(platform[p_ids], dtype=torch.float32, device=device)

        out = model(w, p)

    return out.cpu().numpy()


def prune_model_torch(model, amount=0.4):
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            k = int(param.numel() * amount)
            if k > 0:
                vals, idx = torch.topk(param.abs().flatten(), k, largest=False)
                flat = param.data.flatten()
                flat[idx] = 0
    return model


def quantize_model_simulated(model):
    import copy
    q = copy.deepcopy(model)

    for name, param in q.named_parameters():
        param.data = (torch.round(param.data * 127) / 127)

    return q, None
