import networkx as nx
import pandas as pd

from metabolic_gap_filling.config import PROCESSED_DATA_DIR, RESULTS_DIR


VARIANT = "no_currency_no_biomass"
NEGATIVE_SET = "degree_matched"
METHOD = "node2vec_logistic"
TOP_N = 100


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    graph = nx.read_graphml(PROCESSED_DATA_DIR / f"{VARIANT}_graph.graphml")
    predictions = pd.read_csv(RESULTS_DIR / "node2vec_predictions.csv")
    selected = predictions.loc[
        (predictions["variant"] == VARIANT)
        & (predictions["negative_set"] == NEGATIVE_SET)
        & (predictions["method"] == METHOD)
    ].copy()
    selected = selected.sort_values("score", ascending=False).reset_index(drop=True)
    selected["rank"] = selected.index + 1

    annotated = annotate_edges(selected, graph)
    annotated.to_csv(RESULTS_DIR / "node2vec_ranked_predictions_annotated.csv", index=False)
    annotated.head(TOP_N).to_csv(RESULTS_DIR / "node2vec_top100_predictions.csv", index=False)

    true_positives = annotated.loc[annotated["label"] == 1].copy()
    true_positives.to_csv(RESULTS_DIR / "node2vec_hidden_edge_recovery.csv", index=False)

    reaction_summary = summarize_reactions(true_positives)
    reaction_summary.to_csv(RESULTS_DIR / "node2vec_reaction_recovery_summary.csv", index=False)

    false_positives = annotated.loc[annotated["label"] == 0].head(50)
    false_positives.to_csv(RESULTS_DIR / "node2vec_top_false_positives.csv", index=False)

    print("Wrote annotated top predictions and reaction-level recovery summaries.")
    print("\nTop 20 predictions:")
    columns = ["rank", "metabolite_id", "reaction_id", "label", "score"]
    print(annotated.head(20)[columns].to_string(index=False))
    print("\nTop reaction recovery summaries:")
    print(reaction_summary.head(15).to_string(index=False))


def annotate_edges(predictions: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for row in predictions.itertuples(index=False):
        metabolite_node, reaction_node = metabolite_reaction_order(row.source, row.target)
        metabolite_data = graph.nodes[metabolite_node]
        reaction_data = graph.nodes[reaction_node]
        rows.append(
            {
                **row._asdict(),
                "metabolite_node": metabolite_node,
                "metabolite_id": metabolite_data.get("label", metabolite_node),
                "metabolite_name": metabolite_data.get("name", ""),
                "reaction_node": reaction_node,
                "reaction_id": reaction_data.get("label", reaction_node),
                "reaction_name": reaction_data.get("name", ""),
                "reaction_degree": graph.degree[reaction_node],
                "metabolite_degree": graph.degree[metabolite_node],
                "gene_reaction_rule": reaction_data.get("gene_reaction_rule", ""),
            }
        )
    return pd.DataFrame(rows)


def summarize_reactions(true_edges: pd.DataFrame) -> pd.DataFrame:
    summary = (
        true_edges.groupby(["reaction_id", "reaction_name"], dropna=False)
        .agg(
            hidden_edges=("label", "count"),
            best_rank=("rank", "min"),
            median_rank=("rank", "median"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
            recovered_top10=("rank", lambda values: int((values <= 10).sum())),
            recovered_top50=("rank", lambda values: int((values <= 50).sum())),
            recovered_top100=("rank", lambda values: int((values <= 100).sum())),
        )
        .reset_index()
        .sort_values(["best_rank", "hidden_edges"], ascending=[True, False])
    )
    return summary


def metabolite_reaction_order(source: str, target: str) -> tuple[str, str]:
    if source.startswith("m:") and target.startswith("r:"):
        return source, target
    if source.startswith("r:") and target.startswith("m:"):
        return target, source
    raise ValueError(f"Expected metabolite-reaction edge, got {source}, {target}")


if __name__ == "__main__":
    main()
