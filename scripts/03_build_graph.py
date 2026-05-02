import json

import networkx as nx
import pandas as pd

from metabolic_gap_filling.config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR
from metabolic_gap_filling.data import load_cobra_model
from metabolic_gap_filling.graph import build_bipartite_graph, graph_summary
from metabolic_gap_filling.visualize import degree_table, plot_degree_distribution


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = load_cobra_model(RAW_DATA_DIR / "iJO1366.json")
    graph = build_bipartite_graph(model)

    nx.write_graphml(graph, PROCESSED_DATA_DIR / "iJO1366_bipartite.graphml")

    summary = graph_summary(graph)
    (RESULTS_DIR / "graph_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "graph_summary.csv", index=False)

    degrees = degree_table(graph)
    degrees.to_csv(RESULTS_DIR / "degree_table.csv", index=False)
    plot_degree_distribution(degrees, FIGURES_DIR / "degree_distribution.png")

    print("Built bipartite graph and wrote summary outputs.")


if __name__ == "__main__":
    main()
