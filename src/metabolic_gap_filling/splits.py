from __future__ import annotations

import random
from collections.abc import Iterable

import networkx as nx

from metabolic_gap_filling.graph import METABOLITE, REACTION, nodes_by_type


def make_edge_split(
    graph: nx.Graph,
    test_fraction: float = 0.1,
    random_seed: int = 33,
) -> tuple[nx.Graph, list[tuple[str, str]]]:
    rng = random.Random(random_seed)
    candidate_edges = list(graph.edges())
    rng.shuffle(candidate_edges)

    train_graph = graph.copy()
    test_edges = []
    target_test_edges = round(test_fraction * graph.number_of_edges())

    for u, v in candidate_edges:
        if len(test_edges) >= target_test_edges:
            break
        if train_graph.degree[u] <= 1 or train_graph.degree[v] <= 1:
            continue
        train_graph.remove_edge(u, v)
        test_edges.append((u, v))

    return train_graph, test_edges


def sample_negative_edges(
    graph: nx.Graph,
    n_samples: int,
    excluded_edges: Iterable[tuple[str, str]] = (),
    random_seed: int = 33,
) -> list[tuple[str, str]]:
    rng = random.Random(random_seed)
    metabolites = nodes_by_type(graph, METABOLITE)
    reactions = nodes_by_type(graph, REACTION)
    existing = {_ordered_edge(u, v) for u, v in graph.edges()}
    excluded = {_ordered_edge(u, v) for u, v in excluded_edges}

    negatives = set()
    max_attempts = n_samples * 100
    attempts = 0

    while len(negatives) < n_samples and attempts < max_attempts:
        attempts += 1
        edge = _ordered_edge(rng.choice(metabolites), rng.choice(reactions))
        if edge in existing or edge in excluded:
            continue
        negatives.add(edge)

    if len(negatives) < n_samples:
        raise RuntimeError(f"Only sampled {len(negatives)} negative edges out of {n_samples}")

    return list(negatives)


def sample_degree_matched_negative_edges(
    graph: nx.Graph,
    positive_edges: list[tuple[str, str]],
    excluded_edges: Iterable[tuple[str, str]] = (),
    random_seed: int = 33,
) -> list[tuple[str, str]]:
    rng = random.Random(random_seed)
    metabolites = nodes_by_type(graph, METABOLITE)
    reactions = nodes_by_type(graph, REACTION)
    metabolite_by_degree = _nodes_by_degree(graph, metabolites)
    reaction_by_degree = _nodes_by_degree(graph, reactions)
    existing = {_ordered_edge(u, v) for u, v in graph.edges()}
    excluded = {_ordered_edge(u, v) for u, v in excluded_edges}

    negatives = []
    used = set()
    for edge in positive_edges:
        metabolite, reaction = _ordered_edge(*edge)
        candidate = _sample_matched_edge(
            graph=graph,
            metabolite=metabolite,
            reaction=reaction,
            metabolite_by_degree=metabolite_by_degree,
            reaction_by_degree=reaction_by_degree,
            existing=existing,
            excluded=excluded.union(used),
            rng=rng,
        )
        negatives.append(candidate)
        used.add(candidate)

    return negatives


def _ordered_edge(u: str, v: str) -> tuple[str, str]:
    if u.startswith("m:") and v.startswith("r:"):
        return u, v
    if u.startswith("r:") and v.startswith("m:"):
        return v, u
    return tuple(sorted((u, v)))


def _nodes_by_degree(graph: nx.Graph, nodes: list[str]) -> dict[int, list[str]]:
    grouped = {}
    for node in nodes:
        grouped.setdefault(graph.degree[node], []).append(node)
    return grouped


def _sample_matched_edge(
    graph: nx.Graph,
    metabolite: str,
    reaction: str,
    metabolite_by_degree: dict[int, list[str]],
    reaction_by_degree: dict[int, list[str]],
    existing: set[tuple[str, str]],
    excluded: set[tuple[str, str]],
    rng: random.Random,
) -> tuple[str, str]:
    strategies = [
        ("metabolite", graph.degree[metabolite]),
        ("reaction", graph.degree[reaction]),
    ]
    rng.shuffle(strategies)

    for side, degree in strategies:
        for tolerance in range(0, 8):
            if side == "metabolite":
                candidates = _candidate_nodes_by_degree(metabolite_by_degree, degree, tolerance)
                rng.shuffle(candidates)
                for candidate_metabolite in candidates:
                    candidate = _ordered_edge(candidate_metabolite, reaction)
                    if candidate not in existing and candidate not in excluded:
                        return candidate
            else:
                candidates = _candidate_nodes_by_degree(reaction_by_degree, degree, tolerance)
                rng.shuffle(candidates)
                for candidate_reaction in candidates:
                    candidate = _ordered_edge(metabolite, candidate_reaction)
                    if candidate not in existing and candidate not in excluded:
                        return candidate

    return _sample_any_negative_edge(graph, existing, excluded, rng)


def _candidate_nodes_by_degree(
    nodes_by_degree: dict[int, list[str]],
    degree: int,
    tolerance: int,
) -> list[str]:
    candidates = []
    for candidate_degree in range(max(0, degree - tolerance), degree + tolerance + 1):
        candidates.extend(nodes_by_degree.get(candidate_degree, []))
    return candidates


def _sample_any_negative_edge(
    graph: nx.Graph,
    existing: set[tuple[str, str]],
    excluded: set[tuple[str, str]],
    rng: random.Random,
) -> tuple[str, str]:
    metabolites = nodes_by_type(graph, METABOLITE)
    reactions = nodes_by_type(graph, REACTION)
    for _ in range(10000):
        candidate = _ordered_edge(rng.choice(metabolites), rng.choice(reactions))
        if candidate not in existing and candidate not in excluded:
            return candidate
    raise RuntimeError("Could not sample a fallback negative edge")
