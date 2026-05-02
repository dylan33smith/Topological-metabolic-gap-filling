import networkx as nx
import pandas as pd

from metabolic_gap_filling.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from metabolic_gap_filling.evaluate import binary_ranking_metrics, pr_curve
from metabolic_gap_filling.node2vec_model import (
    fit_edge_classifier,
    fit_node2vec_embeddings,
    make_training_examples,
    predict_edge_scores,
)


VARIANTS = ["base", "no_currency", "no_currency_no_biomass"]
NEGATIVE_SETS = {
    "random": "{variant}_negative_edges.csv",
    "degree_matched": "{variant}_degree_matched_negative_edges.csv",
}
RANDOM_SEED = 33
NODE2VEC_PARAMS = {
    "dimensions": 64,
    "walk_length": 20,
    "num_walks": 10,
    "p": 1.0,
    "q": 1.0,
    "workers": 1,
    "seed": RANDOM_SEED,
}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    prediction_rows = []
    curve_rows = []

    for variant in VARIANTS:
        graph = nx.read_graphml(PROCESSED_DATA_DIR / f"{variant}_train_graph.graphml")
        embeddings = fit_node2vec_embeddings(graph, NODE2VEC_PARAMS)
        train_edges, train_labels = make_training_examples(graph)
        classifier = fit_edge_classifier(embeddings, train_edges, train_labels)
        positive_edges = read_edges(PROCESSED_DATA_DIR / f"{variant}_test_edges.csv")

        for negative_set, filename_template in NEGATIVE_SETS.items():
            negative_edges = read_edges(PROCESSED_DATA_DIR / filename_template.format(variant=variant))
            candidate_edges = positive_edges + negative_edges
            y_true = [1] * len(positive_edges) + [0] * len(negative_edges)
            scores = predict_edge_scores(classifier, embeddings, candidate_edges)
            metrics = binary_ranking_metrics(y_true, scores, k_values=(10, 50, 100))
            metric_rows.append(
                {
                    "variant": variant,
                    "negative_set": negative_set,
                    "method": "node2vec_logistic",
                    **NODE2VEC_PARAMS,
                    **metrics,
                }
            )

            for edge, label, score in zip(candidate_edges, y_true, scores, strict=True):
                prediction_rows.append(
                    {
                        "variant": variant,
                        "negative_set": negative_set,
                        "method": "node2vec_logistic",
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
                        "method": "node2vec_logistic",
                        "point": index,
                        "precision": p_value,
                        "recall": r_value,
                        "threshold": thresholds[index] if index < len(thresholds) else None,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    curves = pd.DataFrame(curve_rows)
    metrics.to_csv(RESULTS_DIR / "node2vec_metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "node2vec_predictions.csv", index=False)
    curves.to_csv(RESULTS_DIR / "node2vec_pr_curves.csv", index=False)

    combined = combine_with_baselines(metrics)
    combined.to_csv(RESULTS_DIR / "model_comparison_metrics.csv", index=False)
    plot_model_comparison(combined, FIGURES_DIR / "model_comparison_average_precision.png")

    print("Wrote Node2Vec metrics, predictions, and comparison outputs.")
    print(combined.sort_values(["negative_set", "variant", "average_precision"]).to_string(index=False))


def read_edges(path) -> list[tuple[str, str]]:
    rows = pd.read_csv(path)
    return list(rows[["source", "target"]].itertuples(index=False, name=None))


def combine_with_baselines(node2vec_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline_metrics = pd.read_csv(RESULTS_DIR / "baseline_metrics.csv")
    common_columns = [
        "variant",
        "negative_set",
        "method",
        "average_precision",
        "auroc",
        "precision_at_10",
        "recall_at_10",
        "precision_at_50",
        "recall_at_50",
        "precision_at_100",
        "recall_at_100",
    ]
    return pd.concat(
        [baseline_metrics[common_columns], node2vec_metrics[common_columns]],
        ignore_index=True,
    )


def plot_model_comparison(metrics: pd.DataFrame, output_path) -> None:
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
        height=4.3,
        aspect=1.25,
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
    figure.figure.subplots_adjust(right=0.80, bottom=0.24, wspace=0.12)
    figure.figure.savefig(output_path, dpi=300)
    plt.close(figure.figure)


if __name__ == "__main__":
    main()
