from __future__ import annotations

import math
import random

import networkx as nx


def random_score(edge: tuple[str, str], random_seed: int = 33) -> float:
    rng = random.Random(f"{random_seed}:{edge[0]}:{edge[1]}")
    return rng.random()


def degree_product_score(graph: nx.Graph, edge: tuple[str, str]) -> float:
    u, v = edge
    return float(graph.degree[u] * graph.degree[v])


def common_neighbors_score(graph: nx.Graph, edge: tuple[str, str]) -> float:
    metabolite_reactions, reaction_context = _reaction_context_sets(graph, edge)
    return float(len(metabolite_reactions.intersection(reaction_context)))


def jaccard_score(graph: nx.Graph, edge: tuple[str, str]) -> float:
    metabolite_reactions, reaction_context = _reaction_context_sets(graph, edge)
    union = metabolite_reactions.union(reaction_context)
    if not union:
        return 0.0
    return float(len(metabolite_reactions.intersection(reaction_context)) / len(union))


def adamic_adar_score(graph: nx.Graph, edge: tuple[str, str]) -> float:
    metabolite_reactions, reaction_context = _reaction_context_sets(graph, edge)
    shared_reactions = metabolite_reactions.intersection(reaction_context)
    score = 0.0
    for reaction in shared_reactions:
        degree = graph.degree[reaction]
        if degree > 1:
            score += 1 / math.log(degree)
    return float(score)


def reaction_context_score(graph: nx.Graph, edge: tuple[str, str]) -> float:
    metabolite, reaction = _metabolite_reaction_order(edge)
    reaction_metabolites = set(graph.neighbors(reaction))
    metabolite_reactions = set(graph.neighbors(metabolite))

    score = 0.0
    for context_metabolite in reaction_metabolites:
        if context_metabolite == metabolite:
            continue
        shared_reactions = metabolite_reactions.intersection(graph.neighbors(context_metabolite))
        score += len(shared_reactions)
    return score


def score_edges(
    graph: nx.Graph,
    edges: list[tuple[str, str]],
    method: str,
    random_seed: int = 33,
) -> list[float]:
    scorers = {
        "random": None,
        "degree_product": degree_product_score,
        "preferential_attachment": degree_product_score,
        "common_neighbors": common_neighbors_score,
        "jaccard": jaccard_score,
        "adamic_adar": adamic_adar_score,
        "reaction_context": reaction_context_score,
    }
    if method not in scorers:
        raise ValueError(f"Unknown baseline method: {method}")
    if method == "random":
        return [random_score(edge, random_seed=random_seed) for edge in edges]
    return [scorers[method](graph, edge) for edge in edges]


def _reaction_context_sets(graph: nx.Graph, edge: tuple[str, str]) -> tuple[set[str], set[str]]:
    metabolite, reaction = _metabolite_reaction_order(edge)
    metabolite_reactions = set(graph.neighbors(metabolite))
    reaction_context = set()

    for context_metabolite in graph.neighbors(reaction):
        if context_metabolite == metabolite:
            continue
        reaction_context.update(graph.neighbors(context_metabolite))

    reaction_context.discard(reaction)
    return metabolite_reactions, reaction_context


def _metabolite_reaction_order(edge: tuple[str, str]) -> tuple[str, str]:
    u, v = edge
    if u.startswith("m:") and v.startswith("r:"):
        return u, v
    if u.startswith("r:") and v.startswith("m:"):
        return v, u
    raise ValueError(f"Expected a metabolite-reaction edge, got {edge}")
