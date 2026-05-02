import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from metabolic_gap_filling.config import FIGURES_DIR, RAW_DATA_DIR, RESULTS_DIR
from metabolic_gap_filling.data import load_cobra_model
from metabolic_gap_filling.filters import KNOWN_CURRENCY_BASE_IDS, base_metabolite_id
from metabolic_gap_filling.filters import PROTECTED_METABOLITE_BASE_IDS
from metabolic_gap_filling.graph import METABOLITE, REACTION, build_bipartite_graph
from metabolic_gap_filling.visualize import degree_table


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = load_cobra_model(RAW_DATA_DIR / "iJO1366.json")
    graph = build_bipartite_graph(model)
    degrees = degree_table(graph)
    degrees["label"] = degrees["node"].str.replace(r"^[mr]:", "", regex=True)
    degrees["base_id"] = degrees["label"].apply(base_metabolite_id)

    summary = summarize_degrees(degrees)
    summary.to_csv(RESULTS_DIR / "network_degree_summary.csv", index=False)

    top_metabolites = top_nodes(degrees, METABOLITE, n=40)
    top_reactions = top_nodes(degrees, REACTION, n=40)
    top_metabolites.to_csv(RESULTS_DIR / "top_metabolites_by_degree.csv", index=False)
    top_reactions.to_csv(RESULTS_DIR / "top_reactions_by_degree.csv", index=False)

    currency_candidates = identify_currency_candidates(degrees)
    currency_candidates.to_csv(RESULTS_DIR / "currency_metabolite_candidates.csv", index=False)

    plot_top_nodes(
        top_metabolites.head(15),
        "Top metabolite degrees",
        FIGURES_DIR / "top_metabolites_by_degree.png",
    )
    plot_top_nodes(
        top_reactions.head(15),
        "Top reaction degrees",
        FIGURES_DIR / "top_reactions_by_degree.png",
    )
    plot_degree_histograms(degrees, FIGURES_DIR / "degree_distribution_by_type.png")
    plot_rank_degree(degrees, FIGURES_DIR / "rank_degree_plot.png")

    print("Wrote network exploration outputs.")
    print(summary.to_string(index=False))
    print("\nTop metabolite candidates:")
    print(top_metabolites.head(10)[["label", "degree"]].to_string(index=False))


def summarize_degrees(degrees: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for node_type, group in degrees.groupby("node_type"):
        values = group["degree"]
        rows.append(
            {
                "node_type": node_type,
                "count": int(values.count()),
                "min": int(values.min()),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "q90": float(values.quantile(0.90)),
                "q95": float(values.quantile(0.95)),
                "q99": float(values.quantile(0.99)),
                "max": int(values.max()),
            }
        )
    return pd.DataFrame(rows)


def top_nodes(degrees: pd.DataFrame, node_type: str, n: int) -> pd.DataFrame:
    columns = ["node", "label", "base_id", "node_type", "degree"]
    return (
        degrees.loc[degrees["node_type"] == node_type, columns]
        .sort_values(["degree", "label"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )


def identify_currency_candidates(degrees: pd.DataFrame) -> pd.DataFrame:
    metabolites = degrees.loc[degrees["node_type"] == METABOLITE].copy()
    degree_cutoff = metabolites["degree"].quantile(0.99)
    metabolites["known_currency_id"] = metabolites["base_id"].isin(KNOWN_CURRENCY_BASE_IDS)
    metabolites["protected_metabolite"] = metabolites["base_id"].isin(PROTECTED_METABOLITE_BASE_IDS)
    metabolites["degree_ge_99th_percentile"] = metabolites["degree"] >= degree_cutoff
    metabolites["selected_for_removal"] = (
        metabolites["known_currency_id"] | metabolites["degree_ge_99th_percentile"]
    ) & ~metabolites["protected_metabolite"]
    candidates = metabolites.loc[
        metabolites["known_currency_id"]
        | metabolites["degree_ge_99th_percentile"]
        | metabolites["protected_metabolite"],
        [
            "node",
            "label",
            "base_id",
            "degree",
            "known_currency_id",
            "degree_ge_99th_percentile",
            "protected_metabolite",
            "selected_for_removal",
        ],
    ]
    return candidates.sort_values(["degree", "label"], ascending=[False, True]).reset_index(drop=True)


def plot_top_nodes(nodes: pd.DataFrame, title: str, output_path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plot_data = nodes.head(12).sort_values("degree", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    sns.barplot(data=plot_data, x="label", y="degree", color="#4C78A8", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Degree")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=7, padding=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_degree_histograms(degrees: pd.DataFrame, output_path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=False)
    for ax, node_type in zip(axes, [METABOLITE, REACTION], strict=True):
        subset = degrees.loc[degrees["node_type"] == node_type].copy()
        max_visible = 15 if node_type == METABOLITE else 10
        subset["degree_bin"] = subset["degree"].apply(
            lambda degree: str(degree) if degree <= max_visible else f">{max_visible}"
        )
        order = [str(value) for value in range(1, max_visible + 1)] + [f">{max_visible}"]
        counts = (
            subset["degree_bin"]
            .value_counts()
            .reindex(order, fill_value=0)
            .rename_axis("degree_bin")
            .reset_index(name="count")
        )
        sns.barplot(data=counts, x="degree_bin", y="count", color="#4C78A8", ax=ax)
        ax.axvline(subset["degree"].median(), color="#E45756", linestyle="--", linewidth=1.5)
        ax.set_title(f"{node_type.title()} degree distribution")
        ax.set_xlabel("Degree bin")
        ax.set_ylabel("Node count")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_rank_degree(degrees: pd.DataFrame, output_path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for node_type, subset in degrees.groupby("node_type"):
        sorted_degrees = subset["degree"].sort_values(ascending=False).reset_index(drop=True)
        ranks = range(1, len(sorted_degrees) + 1)
        ax.plot(ranks, sorted_degrees, marker=".", linestyle="none", markersize=3, label=node_type)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Rank-degree plot")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Degree")
    ax.legend(title="Node type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
