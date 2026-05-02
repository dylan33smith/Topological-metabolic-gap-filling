from __future__ import annotations

import networkx as nx

from metabolic_gap_filling.graph import METABOLITE, REACTION

KNOWN_CURRENCY_BASE_IDS = {
    "h",
    "h2o",
    "atp",
    "adp",
    "amp",
    "pi",
    "ppi",
    "nad",
    "nadh",
    "nadp",
    "nadph",
    "coa",
    "accoa",
    "co2",
    "o2",
    "nh4",
}

PROTECTED_METABOLITE_BASE_IDS = {
    "pyr",
    "accoa",
    "ACP",
}


def select_currency_metabolites(graph: nx.Graph, degree_quantile: float = 0.99) -> list[str]:
    metabolite_nodes = [
        node for node, data in graph.nodes(data=True) if data.get("node_type") == METABOLITE
    ]
    metabolite_degrees = [graph.degree[node] for node in metabolite_nodes]
    degree_cutoff = _quantile(metabolite_degrees, degree_quantile)

    remove_nodes = []
    for node in metabolite_nodes:
        base_id = base_metabolite_id(graph.nodes[node].get("label", node))
        if base_id in PROTECTED_METABOLITE_BASE_IDS:
            continue
        if base_id in KNOWN_CURRENCY_BASE_IDS or graph.degree[node] >= degree_cutoff:
            remove_nodes.append(node)
    return sorted(remove_nodes, key=lambda node: (-graph.degree[node], graph.nodes[node]["label"]))


def select_biomass_reactions(graph: nx.Graph) -> list[str]:
    return sorted(
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == REACTION and data.get("label", "").upper().startswith("BIOMASS")
    )


def remove_currency_metabolites(graph: nx.Graph, degree_quantile: float = 0.99) -> nx.Graph:
    filtered = graph.copy()
    filtered.remove_nodes_from(select_currency_metabolites(filtered, degree_quantile))
    _remove_isolates(filtered)
    return filtered


def remove_biomass_reactions(graph: nx.Graph) -> nx.Graph:
    filtered = graph.copy()
    filtered.remove_nodes_from(select_biomass_reactions(filtered))
    _remove_isolates(filtered)
    return filtered


def _remove_isolates(graph: nx.Graph) -> None:
    graph.remove_nodes_from(list(nx.isolates(graph)))


def base_metabolite_id(label: str) -> str:
    parts = label.split("_")
    if len(parts) > 1 and len(parts[-1]) <= 3:
        return "_".join(parts[:-1])
    return label


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(round(q * (len(sorted_values) - 1)), len(sorted_values) - 1)
    return float(sorted_values[index])
