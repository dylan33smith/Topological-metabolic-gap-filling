import pandas as pd

from metabolic_gap_filling.config import FIGURES_DIR, RESULTS_DIR


VARIANT = "no_currency_no_biomass"
NEGATIVE_SET = "degree_matched"
TOP_VALUES = [10, 50, 100]
METHOD_ORDER = [
    "random",
    "preferential_attachment",
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "reaction_context",
    "node2vec_logistic",
]


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    topn = build_topn_table()
    topn.to_csv(RESULTS_DIR / "topn_prediction_composition.csv", index=False)
    plot_topn(topn, FIGURES_DIR / "topn_prediction_composition.png")
    print("Wrote top-N prediction composition table and figure.")
    print(topn.to_string(index=False))


def build_topn_table() -> pd.DataFrame:
    baseline = pd.read_csv(RESULTS_DIR / "baseline_predictions.csv")
    node2vec = pd.read_csv(RESULTS_DIR / "node2vec_predictions.csv")
    predictions = pd.concat([baseline, node2vec], ignore_index=True)
    predictions = predictions.loc[
        (predictions["variant"] == VARIANT)
        & (predictions["negative_set"] == NEGATIVE_SET)
        & (predictions["method"].isin(METHOD_ORDER))
    ].copy()

    rows = []
    for method in METHOD_ORDER:
        ranked = predictions.loc[predictions["method"] == method].sort_values(
            "score",
            ascending=False,
        )
        for top_n in TOP_VALUES:
            top = ranked.head(top_n)
            positives = int(top["label"].sum())
            negatives = int(len(top) - positives)
            rows.extend(
                [
                    {
                        "variant": VARIANT,
                        "negative_set": NEGATIVE_SET,
                        "method": method,
                        "top_n": top_n,
                        "label_type": "positive",
                        "count": positives,
                        "fraction": positives / top_n,
                    },
                    {
                        "variant": VARIANT,
                        "negative_set": NEGATIVE_SET,
                        "method": method,
                        "top_n": top_n,
                        "label_type": "negative",
                        "count": negatives,
                        "fraction": negatives / top_n,
                    },
                ]
            )
    return pd.DataFrame(rows)


def plot_topn(topn: pd.DataFrame, output_path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    methods = [method for method in METHOD_ORDER if method in set(topn["method"])]
    colors = {"positive": "#4C78A8", "negative": "#E45756"}

    fig, axes = plt.subplots(1, len(TOP_VALUES), figsize=(11.2, 4.8), sharey=False)
    for ax, top_n in zip(axes, TOP_VALUES, strict=True):
        subset = topn.loc[topn["top_n"] == top_n]
        positive_counts = [
            int(
                subset.loc[
                    (subset["method"] == method) & (subset["label_type"] == "positive"),
                    "count",
                ].iloc[0]
            )
            for method in methods
        ]
        negative_counts = [
            int(
                subset.loc[
                    (subset["method"] == method) & (subset["label_type"] == "negative"),
                    "count",
                ].iloc[0]
            )
            for method in methods
        ]

        x = range(len(methods))
        ax.bar(x, positive_counts, color=colors["positive"], label="positive")
        ax.bar(x, negative_counts, bottom=positive_counts, color=colors["negative"], label="negative")
        ax.set_title(f"Top {top_n}")
        ax.set_xticks(list(x))
        ax.set_xticklabels([method.replace("_", " ") for method in methods], rotation=55, ha="right")
        ax.set_xlabel("")
        ax.set_ylim(0, top_n)
        if ax is axes[0]:
            ax.set_ylabel("Prediction count")
        for index, (positives, negatives) in enumerate(zip(positive_counts, negative_counts, strict=True)):
            if positives > 0:
                ax.text(index, positives / 2, str(positives), ha="center", va="center", color="white")
            if negatives > 0:
                ax.text(
                    index,
                    positives + negatives / 2,
                    str(negatives),
                    ha="center",
                    va="center",
                    color="white",
                )

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Actual label", loc="center right")
    fig.suptitle("Composition of top-ranked predictions")
    fig.subplots_adjust(right=0.88, bottom=0.31, top=0.82, wspace=0.15)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
