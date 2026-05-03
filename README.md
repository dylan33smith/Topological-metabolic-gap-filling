# Topological Link Prediction for Metabolic Gap-Filling

This project tests whether graph topology can recover missing metabolic structure in the
*Escherichia coli* K-12 iJO1366 genome-scale metabolic model. The core analysis represents the
model as a bipartite metabolite-reaction graph, evaluates link prediction on hidden
metabolite-reaction edges, and interprets the strongest predictions as gap-filling candidates.


## Project Structure

```text
topological-metabolic-gap-filling/
├── data/
│   ├── raw/              # Downloaded source data, not committed
│   └── processed/        # Derived graph/data files, not committed
├── figures/              # Generated plots for the report
├── report/               # PhysRev
├── results/              # Metrics, model outputs, and ranked predictions
├── scripts/              # Reproducible command-line workflow
├── src/
│   └── metabolic_gap_filling/
└── tests/
```

## Setup


```bash
conda env create -f environment.yml
conda activate topological-gapfill
```


Generic Python setup also works:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

