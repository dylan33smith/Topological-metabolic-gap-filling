from pathlib import Path

import pandas as pd


def load_cobra_model(path: str | Path):
    """Load an SBML or JSON COBRA model from disk."""
    from cobra.io import load_json_model, read_sbml_model

    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_json_model(str(path))
    if path.suffix.lower() in {".xml", ".sbml"}:
        return read_sbml_model(str(path))
    raise ValueError(f"Unsupported model format: {path.suffix}")


def summarize_model(model) -> pd.DataFrame:
    rows = [
        {"component": "reactions", "count": len(model.reactions)},
        {"component": "metabolites", "count": len(model.metabolites)},
        {"component": "genes", "count": len(model.genes)},
    ]
    return pd.DataFrame(rows)


def reaction_table(model) -> pd.DataFrame:
    records = []
    for reaction in model.reactions:
        records.append(
            {
                "reaction_id": reaction.id,
                "name": reaction.name,
                "lower_bound": reaction.lower_bound,
                "upper_bound": reaction.upper_bound,
                "gene_reaction_rule": reaction.gene_reaction_rule,
                "metabolite_count": len(reaction.metabolites),
            }
        )
    return pd.DataFrame(records)
