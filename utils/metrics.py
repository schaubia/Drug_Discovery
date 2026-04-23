"""
utils/metrics.py
Shared accuracy metrics and resource tracking used by all notebooks.
"""

import time
import json
import numpy as np
import psutil
import torch
from pathlib import Path

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr


# ── Accuracy Metrics ────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, prefix=''):
    """
    Compute standard regression metrics for drug activity prediction.
    
    Returns dict with R2, RMSE, MAE, Spearman rho.
    All metrics assume log-transformed IC50 values.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = np.mean(np.abs(y_true - y_pred))
    rho, pval = spearmanr(y_true, y_pred)

    metrics = {
        f'{prefix}r2':      round(r2, 4),
        f'{prefix}rmse':    round(rmse, 4),
        f'{prefix}mae':     round(mae, 4),
        f'{prefix}spearman_rho': round(rho, 4),
        f'{prefix}spearman_p':   round(pval, 6),
    }
    return metrics


def print_metrics(metrics, model_name=''):
    """Pretty print metrics dict."""
    header = f'  {model_name}' if model_name else ''
    print(header)
    for k, v in metrics.items():
        print(f'    {k:<25} {v}')


# ── Resource Tracking ──────────────────────────────────────────────────────

class ResourceTracker:
    """
    Tracks GPU VRAM, CPU RAM, and training time for a model run.
    
    Usage:
        tracker = ResourceTracker('rf_morgan')
        tracker.start()
        # ... train model ...
        stats = tracker.stop()
    """

    def __init__(self, model_name: str, results_dir: str = './results/'):
        self.model_name  = model_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = None
        self._start_vram = None
        self._start_ram  = None

    def _get_vram_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
        return 0.0

    def _get_peak_vram_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
        return 0.0

    def _get_ram_mb(self):
        return psutil.Process().memory_info().rss / 1e6

    def start(self):
        """Call before model training/inference begins."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self._start_time = time.time()
        self._start_vram = self._get_vram_mb()
        self._start_ram  = self._get_ram_mb()
        print(f'[{self.model_name}] Tracking started — '
              f'VRAM: {self._start_vram:.0f} MB, RAM: {self._start_ram:.0f} MB')

    def stop(self):
        """Call after model training/inference completes. Returns stats dict."""
        elapsed = time.time() - self._start_time
        peak_vram  = self._get_peak_vram_mb()
        end_ram    = self._get_ram_mb()
        delta_ram  = end_ram - self._start_ram

        stats = {
            'model':              self.model_name,
            'train_time_sec':     round(elapsed, 2),
            'train_time_min':     round(elapsed / 60, 2),
            'peak_vram_mb':       round(peak_vram, 1),
            'delta_ram_mb':       round(delta_ram, 1),
        }

        # Save to results
        out_path = self.results_dir / f'resource_{self.model_name}.json'
        with open(out_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f'[{self.model_name}] Completed in {elapsed:.1f}s '
              f'| Peak VRAM: {peak_vram:.0f} MB '
              f'| ΔRAM: {delta_ram:.0f} MB')

        return stats


# ── Efficiency Score ────────────────────────────────────────────────────────

def efficiency_score(r2: float, peak_vram_mb: float,
                     train_time_min: float,
                     vram_weight: float = 0.4,
                     time_weight: float = 0.4,
                     accuracy_weight: float = 0.2) -> float:
    """
    Composite efficiency score: accuracy vs. resource cost.

    Higher is better. Normalizes each factor to [0, 1] range
    relative to a reference RTX 3070 budget:
      - VRAM budget:  8000 MB
      - Time budget:  60 minutes

    Formula:
        score = accuracy_weight * R2
              + vram_weight  * (1 - VRAM / budget)
              + time_weight  * (1 - time / budget)
    """
    VRAM_BUDGET = 8000.0   # MB (RTX 3070)
    TIME_BUDGET = 60.0     # minutes

    accuracy_score = max(0.0, r2)
    vram_score     = max(0.0, 1.0 - peak_vram_mb / VRAM_BUDGET)
    time_score     = max(0.0, 1.0 - train_time_min / TIME_BUDGET)

    score = (accuracy_weight  * accuracy_score +
             vram_weight      * vram_score     +
             time_weight      * time_score)

    return round(score, 4)


# ── Results Aggregator ─────────────────────────────────────────────────────

def load_all_results(results_dir: str = './results/') -> dict:
    """
    Load all saved model results for comparison in 04_benchmark.ipynb.
    Returns dict: {model_name: {metrics + resources + efficiency}}
    """
    results_dir = Path(results_dir)
    all_results = {}

    for resource_file in results_dir.glob('resource_*.json'):
        model_name = resource_file.stem.replace('resource_', '')
        with open(resource_file) as f:
            resource_data = json.load(f)

        metrics_file = results_dir / f'metrics_{model_name}.json'
        metrics_data = {}
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics_data = json.load(f)

        combined = {**resource_data, **metrics_data}

        # Compute efficiency score if we have the needed fields
        if 'r2' in combined and 'peak_vram_mb' in combined:
            combined['efficiency_score'] = efficiency_score(
                r2=combined.get('r2', 0),
                peak_vram_mb=combined.get('peak_vram_mb', 8000),
                train_time_min=combined.get('train_time_min', 60),
            )

        all_results[model_name] = combined

    return all_results
