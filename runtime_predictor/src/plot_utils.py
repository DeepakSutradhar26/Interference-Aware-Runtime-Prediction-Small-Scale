import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def plot_true_vs_pred(y_true, y_pred, title, filename):
    ensure_dir(os.path.dirname(filename))
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo,hi],[lo,hi],'r--')
    plt.title(title)
    plt.xlabel("True log-runtime")
    plt.ylabel("Predicted log-runtime")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_heatmap(df, title, filename):
    ensure_dir(os.path.dirname(filename))
    plt.figure(figsize=(10, max(3, len(df)*0.4)))
    sns.heatmap(df, annot=True, cmap="magma", fmt=".4g")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def save_table_as_png(df, title, filename):
    ensure_dir(os.path.dirname(filename))
    fig, ax = plt.subplots(figsize=(10, max(2, len(df)*0.4)))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                   loc="center", cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.title(title)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
