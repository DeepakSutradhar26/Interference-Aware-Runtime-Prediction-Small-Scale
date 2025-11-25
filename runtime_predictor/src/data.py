"""
Synthetic dataset generator for Pitot-style interference model.

Generates:
- workload opcode-like feature vectors (per-workload)
- platform feature vectors (per-platform)
- samples: (workload_id, platform_id, list_of_interferers)
- ground-truth generative Pitot-style parameters used to synthesize log-runtime

Returns train/test splits and metadata needed by the model and plotting.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from .config import (
    RANDOM_SEED,
    N_WORKLOADS,
    N_PLATFORMS,
    MAX_INTERFERERS,
    N_SAMPLES,
    TRAIN_FRACTION,
    LOG_NOISE_STD,
)

def _make_workload_and_platform_features(rng, n_workloads, n_platforms, opcode_dim=24, platform_feat_dim=4):
    # workload opcode-like feature matrix (n_workloads, opcode_dim)
    workload_opcodes = rng.normal(loc=0.5, scale=0.3, size=(n_workloads, opcode_dim)).astype(np.float32)
    workload_opcodes = np.clip(workload_opcodes, 0.0, None)

    # platform features (n_platforms, platform_feat_dim) e.g., cpu_speed, cores, mem, runtime_flag
    platform_feats = rng.normal(loc=1.0, scale=0.2, size=(n_platforms, platform_feat_dim)).astype(np.float32)
    platform_feats = np.clip(platform_feats, 0.01, None)

    return workload_opcodes, platform_feats

def _generate_samples(rng, n_samples, n_workloads, n_platforms, max_interferers):
    # sample workload ids, platform ids, and random interfering sets
    w_ids = rng.integers(0, n_workloads, size=n_samples)
    p_ids = rng.integers(0, n_platforms, size=n_samples)
    interferer_lists = []
    for i in range(n_samples):
        k = int(rng.integers(0, max_interferers+1))
        if k == 0:
            interferer_lists.append([])
        else:
            # ensure interfering workloads are not the same as main workload often, but it's okay
            interferer_lists.append(list(rng.integers(0, n_workloads, size=k)))
    return w_ids, p_ids, interferer_lists

def create_ground_truth(rng, n_workloads, n_platforms, emb_dim=16, n_interf_types=4):
    """
    Create a ground-truth Pitot-like generator: embeddings, baseline scalars, and interference factors.
    These are used to synthesize the y (log-runtime) that the model will try to learn.
    """
    # random true embeddings for workloads and platforms
    true_w_emb = rng.normal(scale=0.8, size=(n_workloads, emb_dim)).astype(np.float32)
    true_p_emb = rng.normal(scale=0.8, size=(n_platforms, emb_dim)).astype(np.float32)

    # baseline scalar per workload and per platform (log space)
    true_w_baseline = rng.normal(loc=0.0, scale=0.3, size=(n_workloads,)).astype(np.float32)
    true_p_baseline = rng.normal(loc=0.0, scale=0.3, size=(n_platforms,)).astype(np.float32)

    # interference factorization per platform:
    # for each platform j, vs_j: (n_interf_types, emb_dim) and vg_j: (n_interf_types, emb_dim)
    true_vs = rng.normal(scale=0.6, size=(n_platforms, n_interf_types, emb_dim)).astype(np.float32)
    true_vg = rng.normal(scale=0.6, size=(n_platforms, n_interf_types, emb_dim)).astype(np.float32)

    return {
        "true_w_emb": true_w_emb,
        "true_p_emb": true_p_emb,
        "true_w_baseline": true_w_baseline,
        "true_p_baseline": true_p_baseline,
        "true_vs": true_vs,
        "true_vg": true_vg,
    }

def synthesize_log_runtime_for_sample(wi, pj, K, gt, noise_std, rng):
    """
    Use ground-truth Pitot generative model to produce one log-runtime.
    Equation:
    logC = wbar_i + pbar_j + w_i^T p_j + sum_t ( w_i^T vs_j[t] * alpha( sum_{k in K} w_k^T vg_j[t] ) ) + noise
    alpha = leaky_relu with slope 0.1
    """
    w_emb = gt["true_w_emb"][wi]
    p_emb = gt["true_p_emb"][pj]
    log_cbar = gt["true_w_baseline"][wi] + gt["true_p_baseline"][pj]

    term = float(np.dot(w_emb, p_emb))

    # interference
    s = gt["true_vs"].shape[1]
    interf_sum = 0.0
    for t in range(s):
        vs_t = gt["true_vs"][pj, t]    # emb_dim
        vg_t = gt["true_vg"][pj, t]
        susceptibility = float(np.dot(w_emb, vs_t))
        magnitude_sum = 0.0
        for k in K:
            magnitude_sum += float(np.dot(gt["true_w_emb"][k], vg_t))
        # alpha = leaky relu
        alpha = magnitude_sum if magnitude_sum >= 0 else 0.1 * magnitude_sum
        interf_sum += susceptibility * alpha

    noise = rng.normal(scale=noise_std)
    logC = float(log_cbar + term + interf_sum + noise)
    return logC

def create_datasets():
    rng = np.random.default_rng(RANDOM_SEED)

    # features for workloads/platforms
    opcode_dim = 24
    platform_feat_dim = 4
    workload_opcodes, platform_feats = _make_workload_and_platform_features(rng, N_WORKLOADS, N_PLATFORMS, opcode_dim, platform_feat_dim)

    # ground-truth generator
    gt = create_ground_truth(rng, N_WORKLOADS, N_PLATFORMS, emb_dim=16, n_interf_types=4)

    # samples
    w_ids, p_ids, interferer_lists = _generate_samples(rng, N_SAMPLES, N_WORKLOADS, N_PLATFORMS, MAX_INTERFERERS)

    # build X structures that include:
    # - workload one-hot (for indexing convenience) OR pass workload id separately to model
    # - platform features
    # but our interference model will accept IDs + side features
    samples = []
    y_log = np.zeros(N_SAMPLES, dtype=np.float32)
    for i in range(N_SAMPLES):
        wi = int(w_ids[i])
        pj = int(p_ids[i])
        K = interferer_lists[i]
        y_log[i] = synthesize_log_runtime_for_sample(wi, pj, K, gt, LOG_NOISE_STD, rng)
        samples.append( (wi, pj, K) )

    # train/test split by sample index
    idx = np.arange(N_SAMPLES)
    train_idx, test_idx = train_test_split(idx, train_size=TRAIN_FRACTION, random_state=RANDOM_SEED, shuffle=True)

    # return everything model needs
    return {
        "samples": samples,
        "y_log": y_log,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "workload_opcodes": workload_opcodes,
        "platform_feats": platform_feats,
        "gt": gt,
    }
