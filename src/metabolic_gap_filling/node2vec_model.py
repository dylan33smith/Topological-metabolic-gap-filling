import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metabolic_gap_filling.splits import sample_negative_edges


def fit_node2vec_embeddings(graph: nx.Graph, params: dict):
    model = Node2Vec(graph, quiet=True, **params)
    return model.fit(window=10, min_count=1, batch_words=128)


def make_training_examples(
    graph: nx.Graph,
    n_negative: int | None = None,
    random_seed: int = 33,
) -> tuple[list[tuple[str, str]], list[int]]:
    positive_edges = list(graph.edges())
    if n_negative is None:
        n_negative = len(positive_edges)
    negative_edges = sample_negative_edges(
        graph,
        n_samples=n_negative,
        excluded_edges=positive_edges,
        random_seed=random_seed,
    )
    return positive_edges + negative_edges, [1] * len(positive_edges) + [0] * len(negative_edges)


def fit_edge_classifier(embeddings, edges: list[tuple[str, str]], labels: list[int], random_seed: int = 33):
    features = edge_features(embeddings, edges)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=random_seed),
    )
    classifier.fit(features, labels)
    return classifier


def predict_edge_scores(classifier, embeddings, edges: list[tuple[str, str]]) -> list[float]:
    features = edge_features(embeddings, edges)
    return classifier.predict_proba(features)[:, 1].tolist()


def edge_features(embeddings, edges: list[tuple[str, str]]) -> np.ndarray:
    return np.asarray([embeddings.wv[u] * embeddings.wv[v] for u, v in edges])
