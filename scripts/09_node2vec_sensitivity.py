import itertools

import networkx as nx
import pandas as pd

from metabolic_gap_filling.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from metabolic_gap_filling.evaluate import binary_ranking_metrics
from metabolic_gap_filling.node2vec_model import (
    fit_edge_classifier,
    fit_node2vec_embeddings,
    make_training_examples,
    predict_edge_scores,
)


VARIANTS = ["no_currency", "no_currency_no_biomass"]
DIMENSIONS = [32, 64, 128]
Q_VALUES = [0.5, 1.0, 2.0]
RANDOM_SEED = 33


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant, dimensions, q_value in itertools.product(VARIANTS, DIMENSIONS, Q_VALUES):
        graph = nx.read_graphml(PROCESSED_DATA_DIR / f"{variant}_train_graph.graphml")
        params = {
            "dimensions": dimensions,
            "walk_length": 20,
            "num_walks": 10,
            "p": 1.0,
            "q": q_value,
            "workers": 1,
            "seed": RANDOM_SEED,
        }
        embeddings = fit_node2vec_embeddings(graph, params)
        train_edges, train_labels = make_training_examples(graph, random_seed=RANDOM_SEED)
        classifier = fit_edge_classifier(embeddings, train_edges, train_labels, random_seed=RANDOM_SEED)

        positive_edges = read_edges(PROCESSED_DATA_DIR / f"{variant}_test_edges.csv")
        negative_edges = read_edges(PROCESSED_DATA_DIR / f"{variant}_degree_matched_negative_edges.csv")
        candidate_edges = positive_edges + negative_edges
        y_true = [1] * len(positive_edges) + [0] * len(negative_edges)
        scores = predict_edge_scores(classifier, embeddings, candidate_edges)
        metrics = binary_ranking_metrics(y_true, scores, k_values=(10, 50, 100))
        rows.append(
            {
                "variant": variant,
                "negative_set": "degree_matched",
                "method": "node2vec_logistic",
                **params,
                **metrics,
            }
        )
        print(
            f"{variant} dim={dimensions} q={q_value}: "
            f"AUPRC={metrics['average_precision']:.4f}, AUROC={metrics['auroc']:.4f}"
        )

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(RESULTS_DIR / "node2vec_sensitivity_metrics.csv", index=False)
    plot_sensitivity(sensitivity, FIGURES_DIR / "node2vec_sensitivity_heatmap.png")
    print("Wrote Node2Vec sensitivity metrics and heatmap.")


def read_edges(path) -> list[tuple[str, str]]:
    rows = pd.read_csv(path)
    return list(rows[["source", "target"]].itertuples(index=False, name=None))


def plot_sensitivity(metrics: pd.DataFrame, output_path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(8.5, 3.8), sharey=True)
    for ax, variant in zip(axes, VARIANTS, strict=True):
        subset = metrics.loc[metrics["variant"] == variant]
        pivot = subset.pivot(index="q", columns="dimensions", values="average_precision")
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=0.5,
            vmax=0.9,
            ax=ax,
            cbar=ax is axes[-1],
        )
        ax.set_title(variant)
        ax.set_xlabel("Dimensions")
        ax.set_ylabel("q" if ax is axes[0] else "")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
