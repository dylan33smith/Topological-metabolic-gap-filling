import json

import networkx as nx
import pandas as pd

from metabolic_gap_filling.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR
from metabolic_gap_filling.data import load_cobra_model
from metabolic_gap_filling.filters import (
    base_metabolite_id,
    remove_biomass_reactions,
    remove_currency_metabolites,
    select_biomass_reactions,
    select_currency_metabolites,
)
from metabolic_gap_filling.graph import build_bipartite_graph, graph_summary
from metabolic_gap_filling.splits import (
    make_edge_split,
    sample_degree_matched_negative_edges,
    sample_negative_edges,
)


TEST_FRACTION = 0.10
RANDOM_SEED = 33


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_cobra_model(RAW_DATA_DIR / "iJO1366.json")
    base_graph = build_bipartite_graph(model)
    no_currency_graph = remove_currency_metabolites(base_graph)
    no_currency_no_biomass_graph = remove_biomass_reactions(no_currency_graph)
    variants = {
        "base": base_graph,
        "no_currency": no_currency_graph,
        "no_currency_no_biomass": no_currency_no_biomass_graph,
    }

    write_removal_records(base_graph, no_currency_graph, no_currency_no_biomass_graph)

    graph_rows = []
    split_rows = []
    for name, graph in variants.items():
        graph_path = PROCESSED_DATA_DIR / f"{name}_graph.graphml"
        nx.write_graphml(graph, graph_path)

        graph_row = {"variant": name, **graph_summary(graph)}
        graph_rows.append(graph_row)

        train_graph, test_edges = make_edge_split(
            graph,
            test_fraction=TEST_FRACTION,
            random_seed=RANDOM_SEED,
        )
        negative_edges = sample_negative_edges(
            graph,
            n_samples=len(test_edges),
            excluded_edges=test_edges,
            random_seed=RANDOM_SEED,
        )
        degree_matched_negative_edges = sample_degree_matched_negative_edges(
            graph,
            positive_edges=test_edges,
            excluded_edges=test_edges,
            random_seed=RANDOM_SEED,
        )

        nx.write_graphml(train_graph, PROCESSED_DATA_DIR / f"{name}_train_graph.graphml")
        write_edges(test_edges, PROCESSED_DATA_DIR / f"{name}_test_edges.csv")
        write_edges(negative_edges, PROCESSED_DATA_DIR / f"{name}_negative_edges.csv")
        write_edges(
            degree_matched_negative_edges,
            PROCESSED_DATA_DIR / f"{name}_degree_matched_negative_edges.csv",
        )

        split_rows.append(
            {
                "variant": name,
                "test_fraction": TEST_FRACTION,
                "random_seed": RANDOM_SEED,
                "train_edges": train_graph.number_of_edges(),
                "test_positive_edges": len(test_edges),
                "test_negative_edges": len(negative_edges),
                "degree_matched_negative_edges": len(degree_matched_negative_edges),
                "train_nodes": train_graph.number_of_nodes(),
            }
        )

    pd.DataFrame(graph_rows).to_csv(RESULTS_DIR / "graph_variant_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(RESULTS_DIR / "benchmark_split_summary.csv", index=False)
    (RESULTS_DIR / "benchmark_split_config.json").write_text(
        json.dumps({"test_fraction": TEST_FRACTION, "random_seed": RANDOM_SEED}, indent=2)
    )

    print("Prepared graph variants and benchmark splits.")
    print(pd.DataFrame(graph_rows).to_string(index=False))
    print()
    print(pd.DataFrame(split_rows).to_string(index=False))


def write_edges(edges: list[tuple[str, str]], output_path) -> None:
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(output_path, index=False)


def write_removal_records(base_graph, no_currency_graph, no_currency_no_biomass_graph) -> None:
    currency_nodes = select_currency_metabolites(base_graph)
    biomass_nodes = select_biomass_reactions(no_currency_graph)

    no_currency_removed = set(base_graph.nodes) - set(no_currency_graph.nodes)
    no_currency_no_biomass_removed = set(no_currency_graph.nodes) - set(no_currency_no_biomass_graph.nodes)

    currency_records = [
        node_record(base_graph, node, "selected_currency_metabolite") for node in currency_nodes
    ]
    biomass_records = [
        node_record(no_currency_graph, node, "selected_biomass_reaction") for node in biomass_nodes
    ]

    no_currency_records = []
    for node in sorted(no_currency_removed):
        reason = "selected_currency_metabolite" if node in currency_nodes else "isolated_after_currency_removal"
        no_currency_records.append(node_record(base_graph, node, reason))

    no_currency_no_biomass_records = []
    for node in sorted(no_currency_no_biomass_removed):
        reason = "selected_biomass_reaction" if node in biomass_nodes else "isolated_after_biomass_removal"
        no_currency_no_biomass_records.append(node_record(no_currency_graph, node, reason))

    pd.DataFrame(currency_records).to_csv(RESULTS_DIR / "removed_currency_metabolites.csv", index=False)
    pd.DataFrame(biomass_records).to_csv(RESULTS_DIR / "removed_biomass_reactions.csv", index=False)
    pd.DataFrame(no_currency_records).to_csv(RESULTS_DIR / "removed_nodes_no_currency.csv", index=False)
    pd.DataFrame(no_currency_no_biomass_records).to_csv(
        RESULTS_DIR / "removed_nodes_no_currency_no_biomass.csv",
        index=False,
    )


def node_record(graph, node: str, reason: str) -> dict:
    data = graph.nodes[node]
    label = data.get("label", node)
    return {
        "node": node,
        "label": label,
        "base_id": base_metabolite_id(label) if node.startswith("m:") else label,
        "node_type": data.get("node_type", ""),
        "degree_before_removal": graph.degree[node],
        "removal_reason": reason,
    }


if __name__ == "__main__":
    main()
