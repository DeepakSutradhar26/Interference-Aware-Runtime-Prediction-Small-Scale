# Small-Scale Pitot-Style Runtime Predictor

This project is a **simplified re-creation** of the method used in  
“Interference-Aware Edge Runtime Prediction with Conformal Matrix Completion (Pitot)”.

It:
- Predicts **runtime (log-runtime) with regression**, like the paper.
- Uses a **neural network regression model** (scikit-learn `MLPRegressor`).
- Generates a **synthetic workload/platform/interference dataset**.
- Implements:
  - Baseline model
  - Pruning
  - Quantization (8-bit, simulated)
  - Pruning + Quantization
- Produces **accuracy plots** comparing all four variants.

No PyTorch or TensorFlow are used.

## 1. Install

```bash
cd runtime-predictor

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
