# Topological Link Prediction for Metabolic Gap-Filling

This project tests whether graph topology can recover missing metabolic structure in the
*Escherichia coli* K-12 iJO1366 genome-scale metabolic model. The core analysis represents the
model as a bipartite metabolite-reaction graph, evaluates link prediction on hidden
metabolite-reaction edges, and interprets the strongest predictions as gap-filling candidates.

The project is being built for a graduate networks course, so NetworkX is the primary graph
library for construction, metrics, and baseline network analysis.

## Project Structure

```text
topological-metabolic-gap-filling/
├── data/
│   ├── raw/              # Downloaded source data, not committed
│   └── processed/        # Derived graph/data files, not committed
├── docs/                 # Living notes for decisions, results, and writeup material
├── figures/              # Generated plots for the report
├── notebooks/            # Optional exploratory notebooks
├── report/               # PhysRev/RevTeX report skeleton
├── results/              # Metrics, model outputs, and ranked predictions
├── scripts/              # Reproducible command-line workflow
├── src/
│   └── metabolic_gap_filling/
└── tests/
```

## Setup

The working environment for this project is a WSL conda environment named `topological-gapfill`.

From WSL:

```bash
cd "/mnt/c/Users/dylan/OneDrive/Documents/New project 2/topological-metabolic-gap-filling"
conda env create -f environment.yml
conda activate topological-gapfill
```

If the environment already exists, activate it and update the editable install:

```bash
cd "/mnt/c/Users/dylan/OneDrive/Documents/New project 2/topological-metabolic-gap-filling"
conda activate topological-gapfill
python -m pip install -e ".[dev]"
```

Generic Python setup also works:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Planned Workflow

1. Download iJO1366 from BiGG Models.
2. Load the model with COBRApy and verify reaction, metabolite, and gene counts.
3. Build a NetworkX bipartite metabolite-reaction graph.
4. Compute exploratory network metrics and identify high-degree currency metabolites.
5. Create controlled train/test splits by hiding metabolite-reaction edges.
6. Evaluate random and topology-based baselines.
7. Train Node2Vec embeddings and evaluate link prediction performance.
8. Aggregate link scores into reaction-level candidate predictions.
9. Optionally validate top predictions using KEGG, UniProt, BioCyc/EcoCyc, or BLAST evidence.
10. Write the final 4-6 page PhysRev-style report.

## Current Commands

```bash
conda activate topological-gapfill
python scripts/01_download_bigg_model.py
python scripts/02_summarize_model.py
python scripts/03_build_graph.py
python scripts/04_explore_network.py
python scripts/05_prepare_benchmark_splits.py
python scripts/06_make_removal_slider.py
python scripts/07_run_baselines.py
python scripts/08_run_node2vec.py
python scripts/09_node2vec_sensitivity.py
python scripts/10_analyze_top_predictions.py
python scripts/11_make_final_figures.py
python scripts/12_make_topn_prediction_figure.py
pytest
```

## Current Status

Initial scaffold, WSL conda environment, data download, model summary, and first NetworkX graph
build are complete. See `docs/project_log.md` for the living record of what has been done, what we
have learned, and what should feed into the report.
