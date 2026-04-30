# Drug Discovery — Model Benchmark & Optimization Pipeline
**RTX 3070 | EGFR Kinase | Open Source**

Inspired by GPT-Rosalind (OpenAI, April 2026) — a local, transparent,
open-source alternative for comparing and optimizing drug discovery models.

---

## Goal
Compare 4 open-source models on EGFR kinase inhibition prediction,
measure accuracy AND resource cost (VRAM, time, CO2), pick the best,
then optimize it for maximum efficiency on a consumer GPU.

---

## Notebook Order

| Notebook | What it does | Status |
|---|---|---|
| `00_setup.ipynb` | Install deps, verify RTX 3070, configure AMP | ✓ Ready |
| `01_data.ipynb` | Load EGFR from TDC, clean, split, EDA | ✓ Ready |
| `02_baselines.ipynb` | Train 4 models, log accuracy + resources | 🔜 Next |
| `03_comparison.ipynb` | Benchmark table, efficiency score, pick winner | 🔜 |
| `04_optimize.ipynb` | Apply AMP, pruning, distillation to winner | 🔜 |
| `05_generate.ipynb` | Use optimized model to score/generate candidates | 🔜 |

---

## Models Compared

| Model | Type | Expected VRAM |
|---|---|---|
| RF + Morgan FP | Classical ML | < 100 MB (CPU) |
| AttentiveFP | Graph attention | ~1.2 GB |
| MPNN (DeepChem) | Message passing net | ~2.5 GB |
| ChemBERTa-2 | Pretrained transformer | ~5.0 GB |

---

## Optimization Techniques (04_optimize.ipynb)

1. **Mixed precision (AMP)** — already enabled from setup
2. **Gradient checkpointing** — trade compute for memory
3. **Knowledge distillation** — smaller student model
4. **INT8 quantization** — 4x inference memory reduction
5. **Optuna HPO** — find smallest model hitting accuracy target

---

## Environment

```cmd
Step 1 — Create the environment (in your new cmd/Anaconda Prompt):
conda create -n drug_discovery python=3.10
This will ask Proceed ([y]/n)? — type y and wait ~2 minutes.
Step 2 — Activate it:
conda activate drug_discovery
You should see (drug_discovery) appear at the start of the prompt line.
Step 3 — Install Jupyter inside the environment:
pip install jupyter ipykernel
python -m ipykernel install --user --name drug_discovery --display-name "drug_discovery"
Step 4 — Install PyTorch with CUDA for your RTX 3070 (do this before the notebook, it's the heaviest install and needs the right CUDA version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
This will download ~2.5GB — normal, just wait for it.
Step 5 — Navigate to your project folder and launch Jupyter:
cd "D:\courses\ML\My projects\Applications\Drug_Discovery"
jupyter notebook
Step 6 — THEN open 00_setup.ipynb in the browser that pops up, make sure the kernel in the top-right says drug_discovery, and run it top to bottom.     
```

---

## Hardware Target
- GPU: NVIDIA RTX 3070 (8GB VRAM)
- All models fit within VRAM budget with AMP enabled
- Energy tracked via CodeCarbon (country: Bulgaria / BGR)

---

## Benchmark Reference
GPT-Rosalind (OpenAI, April 2026) public BixBench scores will be
added to `04_benchmark.ipynb` as a comparison ceiling when published.
