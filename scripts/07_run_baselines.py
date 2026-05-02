import networkx as nx
import pandas as pd

from metabolic_gap_filling.baselines import score_edges
from metabolic_gap_filling.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from metabolic_gap_filling.evaluate import binary_ranking_metrics, pr_curve


VARIANTS = ["base", "no_currency", "no_currency_no_biomass"]
NEGATIVE_SETS = {
    "random": "{variant}_negative_edges.csv",
    "degree_matched": "{variant}_degree_matched_negative_edges.csv",
}
METHODS = [
    "random",
    "preferential_attachment",
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "reaction_context",
]
RANDOM_SEED = 33


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    prediction_rows = []
    curve_rows = []

    for variant in VARIANTS:
        graph = nx.read_graphml(PROCESSED_DATA_DIR / f"{variant}_train_graph.graphml")
        positive_edges = read_edges(PROCESSED_DATA_DIR / f"{variant}_test_edges.csv")

        for negative_set, filename_template in NEGATIVE_SETS.items():
            negative_edges = read_edges(PROCESSED_DATA_DIR / filename_template.format(variant=variant))
            candidate_edges = positive_edges + negative_edges
            y_true = [1] * len(positive_edges) + [0] * len(negative_edges)

            for method in METHODS:
                scores = score_edges(
                    graph,
                    candidate_edges,
                    method=method,
                    random_seed=RANDOM_SEED,
                )
                metrics = binary_ranking_metrics(y_true, scores, k_values=(10, 50, 100))
                metric_rows.append(
                    {
                        "variant": variant,
                        "negative_set": negative_set,
                        "method": method,
                        **metrics,
                    }
                )

                for edge, label, score in zip(candidate_edges, y_true, scores, strict=True):
                    prediction_rows.append(
                        {
                            "variant": variant,
                            "negative_set": negative_set,
                            "method": method,
                            "source": edge[0],
                            "target": edge[1],
                            "label": label,
                            "score": score,
                        }
                    )

                precision, recall, thresholds = pr_curve(y_true, scores)
                for index, (p_value, r_value) in enumerate(zip(precision, recall, strict=True)):
                    curve_rows.append(
                        {
                            "variant": variant,
                            "negative_set": negative_set,
                            "method": method,
                            "point": index,
                            "precision": p_value,
                            "recall": r_value,
                            "threshold": thresholds[index] if index < len(thresholds) else None,
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    curves = pd.DataFrame(curve_rows)

    metrics.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "baseline_predictions.csv", index=False)
    curves.to_csv(RESULTS_DIR / "baseline_pr_curves.csv", index=False)
    plot_baseline_metrics(metrics, FIGURES_DIR / "baseline_average_precision.png")

    sorted_metrics = metrics.sort_values(
        ["variant", "negative_set", "average_precision"],
        ascending=[True, True, False],
    )
    print("Wrote baseline metrics, predictions, and precision-recall curves.")
    print(sorted_metrics.to_string(index=False))


def read_edges(path) -> list[tuple[str, str]]:
    rows = pd.read_csv(path)
    return list(rows[["source", "target"]].itertuples(index=False, name=None))


def plot_baseline_metrics(metrics: pd.DataFrame, output_path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    plot_data = metrics.copy()
    plot_data["method"] = plot_data["method"].str.replace("_", " ")
    plot_data["negative_set"] = plot_data["negative_set"].str.replace("_", " ")

    figure = sns.catplot(
        data=plot_data,
        kind="bar",
        x="variant",
        y="average_precision",
        hue="method",
        col="negative_set",
        height=4.2,
        aspect=1.2,
        sharey=True,
        legend_out=True,
    )
    figure.set_axis_labels("", "Average precision / AUPRC")
    figure.set_titles("{col_name} negatives")
    for ax in figure.axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, 1)
    if figure.legend is not None:
        figure.legend.set_title("Method")
    figure.figure.subplots_adjust(right=0.82, bottom=0.25, wspace=0.12)
    figure.figure.savefig(output_path, dpi=300)
    plt.close(figure.figure)


if __name__ == "__main__":
    main()
