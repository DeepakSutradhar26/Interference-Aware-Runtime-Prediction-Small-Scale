import os
import time
import numpy as np
import torch
import pandas as pd

from .data import create_datasets
from .model import create_model, train_model, predict, prune_model, quantize_model
from .statistics import StatisticsTracker
from .hardware_profiler import HardwareProfiler
from .plot_utils import plot_true_vs_pred, plot_heatmap


RESULTS_DIR = "results"
PRUNE_FRACTION = 0.4


def run_all_experiments():

    for sub in ["plots", "statistics", "hardware", "tables", "models"]:
        os.makedirs(os.path.join(RESULTS_DIR, sub), exist_ok=True)

    print("Generating synthetic dataset…")
    dataset = create_datasets()

    y_log = dataset["y_log"]
    train_idx = dataset["train_idx"]
    test_idx  = dataset["test_idx"]

    w_dim = dataset["workload_opcodes"].shape[1]
    p_dim = dataset["platform_feats"].shape[1]

    model = create_model(w_dim, p_dim)
    stats = StatisticsTracker(RESULTS_DIR)
    hw = HardwareProfiler(RESULTS_DIR)

    # ----------------------------------------------------------
    # ⭐ Famous and standard epoch count = 100
    # ----------------------------------------------------------
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 5e-4

    print(f"Training model for {EPOCHS} epochs…")
    model = train_model(model, dataset, lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # ----------------------------------------------------------
    # Test set evaluation
    # ----------------------------------------------------------
    pred = predict(model, dataset, test_idx)
    true = y_log[test_idx]

    mape = np.mean(np.abs((pred - true) / true))
    acc = (1 - mape) * 100
    r2 = 1 - np.sum((pred - true)**2) / np.sum((true - np.mean(true))**2)

    pd.DataFrame({
        "MAPE":[mape],
        "Accuracy%":[acc],
        "R2":[r2]
    }).to_csv(os.path.join(RESULTS_DIR,"statistics","final_baseline.csv"), index=False)

    plot_true_vs_pred(true, pred,
                      "True vs Predicted",
                      os.path.join(RESULTS_DIR,"plots","true_vs_pred.png"))

    # ----------------------------------------------------------
    # Hardware profiling
    # ----------------------------------------------------------
    batches = int(np.ceil(len(train_idx) / BATCH_SIZE))
    hw.log_training(model, EPOCHS, batches, w_dim, p_dim)
    hw.log_inference(model, w_dim, p_dim)

    # ----------------------------------------------------------
    # Pruning + Quantization
    # ----------------------------------------------------------
    print("\nPruning...")
    pr = prune_model(model, PRUNE_FRACTION)

    print("Quantizing...")
    q, _ = quantize_model(pr)

    pred_pr = predict(pr, dataset, test_idx)
    pred_q  = predict(q, dataset, test_idx)

    m_pr = np.mean(np.abs((pred_pr - true) / true))
    m_q  = np.mean(np.abs((pred_q  - true) / true))

    df = pd.DataFrame({
        "Accuracy%":[acc, (1 - m_pr)*100, (1 - m_q)*100]
    }, index=["Baseline","Pruned","Quantized"])

    plot_heatmap(df, "Model Comparison (Accuracy %)",
                 os.path.join(RESULTS_DIR,"tables","model_comparison_heatmap.png"))

    print("\nDONE. Check results/ folder.")
