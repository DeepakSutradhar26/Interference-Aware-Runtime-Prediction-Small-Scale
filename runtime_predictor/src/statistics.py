import os
import json
import numpy as np
import pandas as pd
from .plot_utils import plot_heatmap, save_table_as_png

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

class StatisticsTracker:
    def __init__(self, results_root):
        self.stats_dir = os.path.join(results_root, "statistics")
        self.tables_dir = os.path.join(results_root, "tables")
        ensure_dir(self.stats_dir)
        ensure_dir(self.tables_dir)

        self.loss = []
        self.acc = []
        self.mape = []
        self.r2 = []
        self.time = []

        self.conv = {
            "MAPE<0.2": None,
            "Loss<0.05": None,
            "R2>0.8": None,
            "FullEpoch": None
        }

    # ----------------------------------------------------
    def record(self, epoch, loss, acc, mape, r2, t):
        self.loss.append(loss)
        self.acc.append(acc)
        self.mape.append(mape)
        self.r2.append(r2)
        self.time.append(t)

        # Convergence detection
        if self.conv["MAPE<0.2"] is None and mape < 0.2:
            self.conv["MAPE<0.2"] = epoch

        if self.conv["Loss<0.05"] is None and loss < 0.05:
            self.conv["Loss<0.05"] = epoch

        if self.conv["R2>0.8"] is None and r2 > 0.8:
            self.conv["R2>0.8"] = epoch

        # Full epoch convergence (filled at end)

    # ----------------------------------------------------
    def finalize(self, total_epochs):
        if self.conv["FullEpoch"] is None:
            self.conv["FullEpoch"] = total_epochs

        # SAVE CSVs
        pd.DataFrame({"epoch":range(1,len(self.loss)+1),"loss":self.loss}).to_csv(
            os.path.join(self.stats_dir,"loss_per_epoch.csv"), index=False)
        pd.DataFrame({"epoch":range(1,len(self.acc)+1),"accuracy":self.acc}).to_csv(
            os.path.join(self.stats_dir,"accuracy_per_epoch.csv"), index=False)
        pd.DataFrame({"epoch":range(1,len(self.mape)+1),"mape":self.mape}).to_csv(
            os.path.join(self.stats_dir,"mape_per_epoch.csv"), index=False)
        pd.DataFrame({"epoch":range(1,len(self.r2)+1),"r2":self.r2}).to_csv(
            os.path.join(self.stats_dir,"r2_per_epoch.csv"), index=False)
        pd.DataFrame({"epoch":range(1,len(self.time)+1),"time":self.time}).to_csv(
            os.path.join(self.stats_dir,"time_per_epoch.csv"), index=False)

        # SAVE HEATMAPS
        plot_heatmap(pd.DataFrame({"loss":self.loss}), "Loss per Epoch",
                     os.path.join(self.tables_dir,"loss_heatmap.png"))

        plot_heatmap(pd.DataFrame({"accuracy":self.acc}), "Accuracy per Epoch",
                     os.path.join(self.tables_dir,"accuracy_heatmap.png"))

        plot_heatmap(pd.DataFrame({"mape":self.mape}), "MAPE per Epoch",
                     os.path.join(self.tables_dir,"mape_heatmap.png"))

        plot_heatmap(pd.DataFrame({"r2":self.r2}), "R2 per Epoch",
                     os.path.join(self.tables_dir,"r2_heatmap.png"))

        # SAVE convergence JSON
        with open(os.path.join(self.stats_dir,"convergence.json"),"w") as f:
            json.dump(self.conv, f, indent=4)

        # SAVE convergence table PNG
        df = pd.DataFrame.from_dict(self.conv, orient="index", columns=["epoch"])
        save_table_as_png(df, "Convergence Table",
                          os.path.join(self.tables_dir,"convergence_table.png"))
