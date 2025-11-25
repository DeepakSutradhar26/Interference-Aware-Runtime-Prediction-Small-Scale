import os
import time
import json
import torch
import numpy as np
import pandas as pd
from .plot_utils import plot_heatmap, save_table_as_png

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

class HardwareProfiler:
    def __init__(self, results_root):
        self.hw_dir = os.path.join(results_root, "hardware")
        self.tables_dir = os.path.join(results_root, "tables")
        ensure_dir(self.hw_dir)
        ensure_dir(self.tables_dir)

    # ----------------------------------------------------
    def estimate_flops(self, model, w_dim, p_dim):
        """
        Very simple FLOPs estimator.
        """
        fwd = 0
        for name,layer in model.named_modules():
            if isinstance(layer, torch.nn.Linear):
                in_f = layer.in_features
                out_f = layer.out_features
                fwd += in_f * out_f * 2
        bwd = fwd * 2
        return fwd, bwd, fwd+bwd

    # ----------------------------------------------------
    def log_training(self, model, epochs, batches, w_dim, p_dim):
        fwd, bwd, it_flops = self.estimate_flops(model, w_dim, p_dim)

        total_flops = epochs * batches * it_flops
        energy = total_flops * 4e-12

        df = pd.DataFrame({
            "metric":["epochs","batches","fwd_flops","bwd_flops","it_flops","total_flops","energy(J)"],
            "value":[epochs,batches,fwd,bwd,it_flops,total_flops,energy]
        }).set_index("metric")

        df.to_csv(os.path.join(self.hw_dir,"hardware_summary.csv"))
        save_table_as_png(df, "Hardware Summary",
                          os.path.join(self.tables_dir,"hardware_summary.png"))
        plot_heatmap(df, "Hardware Summary Heatmap",
                     os.path.join(self.tables_dir,"hardware_summary_heatmap.png"))

    # ----------------------------------------------------
    def log_inference(self, model, w_dim, p_dim):
        fwd, bwd, it_flops = self.estimate_flops(model, w_dim, p_dim)
        df = pd.DataFrame({"metric":["inference_flops"],"value":[fwd]}).set_index("metric")
        df.to_csv(os.path.join(self.hw_dir,"hardware_inference.csv"))
        save_table_as_png(df,"Inference FLOPs",os.path.join(self.tables_dir,"inference_flops.png"))
