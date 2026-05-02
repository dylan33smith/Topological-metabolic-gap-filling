from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_degree_distribution(degree_table: pd.DataFrame, output_path: str | Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(
        data=degree_table,
        x="degree",
        hue="node_type",
        bins=40,
        element="step",
        stat="count",
        common_norm=False,
        ax=ax,
    )
    ax.set_xlabel("Degree")
    ax.set_ylabel("Node count")
    ax.set_title("Bipartite graph degree distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def degree_table(graph) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node": node,
            "node_type": data.get("node_type", "unknown"),
            "degree": graph.degree[node],
        }
        for node, data in graph.nodes(data=True)
    )
