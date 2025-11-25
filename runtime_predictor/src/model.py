from .interference_model import (
    PitotModel,
    train_pitot,
    predict_pitot,
    prune_model_torch,
    quantize_model_simulated
)
from .config import N_WORKLOADS, N_PLATFORMS
import torch


def create_model(workload_feat_dim, platform_feat_dim):
    return PitotModel(
        n_workloads=N_WORKLOADS,
        n_platforms=N_PLATFORMS,
        workload_feat_dim=workload_feat_dim,
        platform_feat_dim=platform_feat_dim
    )


def train_model(model, dataset, lr=1e-3, batch_size=128, epochs=40,
                device=torch.device("cpu")):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    return train_pitot(model, dataset, optimizer,
                       batch_size=batch_size, epochs=epochs, device=device)


def predict(model, dataset, indices, device=torch.device("cpu")):
    return predict_pitot(model, dataset, indices, device=device)


def prune_model(model, amount):
    return prune_model_torch(model, amount)


def quantize_model(model):
    return quantize_model_simulated(model)
