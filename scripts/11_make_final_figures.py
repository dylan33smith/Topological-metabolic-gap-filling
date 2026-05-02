import pandas as pd

from metabolic_gap_filling.config import FIGURES_DIR, RESULTS_DIR


MAIN_VARIANT = "no_currency_no_biomass"
MAIN_NEGATIVE_SET = "degree_matched"
MAIN_METHODS = ["jaccard", "adamic_adar", "reaction_context", "node2vec_logistic"]


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_main_pr_curves()
    plot_top_reaction_recovery()
    plot_filtered_model_comparison()
    print("Wrote final report figure candidates.")


def plot_main_pr_curves() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    baseline = pd.read_csv(RESULTS_DIR / "baseline_pr_curves.csv")
    node2vec = pd.read_csv(RESULTS_DIR / "node2vec_pr_curves.csv")
    curves = pd.concat([baseline, node2vec], ignore_index=True)
    curves = curves.loc[
        (curves["variant"] == MAIN_VARIANT)
        & (curves["negative_set"] == MAIN_NEGATIVE_SET)
        & (curves["method"].isin(MAIN_METHODS))
    ].copy()
    curves["method"] = curves["method"].str.replace("_", " ")

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    sns.lineplot(data=curves, x="recall", y="precision", hue="method", ax=ax)
    ax.set_title("Precision-recall curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_precision_recall_curves.png", dpi=300)
    plt.close(fig)


def plot_top_reaction_recovery() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    reactions = pd.read_csv(RESULTS_DIR / "node2vec_reaction_recovery_summary.csv").head(12)
    reactions = reactions.sort_values("best_rank", ascending=False)

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.barplot(data=reactions, x="best_rank", y="reaction_id", color="#4C78A8", ax=ax)
    ax.set_title("Best-ranked recovered hidden edges by reaction")
    ax.set_xlabel("Best rank among candidate edges")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_reaction_recovery.png", dpi=300)
    plt.close(fig)


def plot_filtered_model_comparison() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics = pd.read_csv(RESULTS_DIR / "model_comparison_metrics.csv")
    metrics = metrics.loc[
        (metrics["variant"].isin(["no_currency", "no_currency_no_biomass"]))
        & (metrics["negative_set"] == MAIN_NEGATIVE_SET)
    ].copy()
    metrics["method"] = metrics["method"].str.replace("_", " ")

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    sns.barplot(data=metrics, x="variant", y="average_precision", hue="method", ax=ax)
    ax.set_title("Filtered graph performance with degree-matched negatives")
    ax.set_xlabel("")
    ax.set_ylabel("Average precision / AUPRC")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=12)
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_filtered_model_comparison.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
