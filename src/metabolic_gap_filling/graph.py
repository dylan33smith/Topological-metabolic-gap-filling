import networkx as nx

METABOLITE = "metabolite"
REACTION = "reaction"


def build_bipartite_graph(model, include_exchange: bool = False) -> nx.Graph:
    graph = nx.Graph()

    for metabolite in model.metabolites:
        graph.add_node(
            f"m:{metabolite.id}",
            bipartite=METABOLITE,
            node_type=METABOLITE,
            label=metabolite.id,
            name=getattr(metabolite, "name", ""),
            compartment=getattr(metabolite, "compartment", ""),
        )

    for reaction in model.reactions:
        if not include_exchange and _is_boundary_reaction(reaction):
            continue

        reaction_node = f"r:{reaction.id}"
        graph.add_node(
            reaction_node,
            bipartite=REACTION,
            node_type=REACTION,
            label=reaction.id,
            name=getattr(reaction, "name", ""),
            lower_bound=float(reaction.lower_bound),
            upper_bound=float(reaction.upper_bound),
            reversible=bool(reaction.reversibility),
            gene_reaction_rule=getattr(reaction, "gene_reaction_rule", ""),
        )

        for metabolite, coefficient in reaction.metabolites.items():
            metabolite_node = f"m:{metabolite.id}"
            role = "substrate" if coefficient < 0 else "product"
            graph.add_edge(
                metabolite_node,
                reaction_node,
                coefficient=float(coefficient),
                role=role,
            )

    return graph


def graph_summary(graph: nx.Graph) -> dict:
    metabolite_nodes = nodes_by_type(graph, METABOLITE)
    reaction_nodes = nodes_by_type(graph, REACTION)
    components = list(nx.connected_components(graph))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "metabolite_nodes": len(metabolite_nodes),
        "reaction_nodes": len(reaction_nodes),
        "connected_components": len(components),
        "largest_component_nodes": max((len(c) for c in components), default=0),
        "density": nx.density(graph),
    }


def nodes_by_type(graph: nx.Graph, node_type: str) -> list[str]:
    return [node for node, data in graph.nodes(data=True) if data.get("node_type") == node_type]


def _is_boundary_reaction(reaction) -> bool:
    boundary = getattr(reaction, "boundary", False)
    if boundary:
        return True
    reaction_id = reaction.id.upper()
    return reaction_id.startswith(("EX_", "DM_", "SK_"))
