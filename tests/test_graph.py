from metabolic_gap_filling.graph import METABOLITE, REACTION, build_bipartite_graph, nodes_by_type


class DummyMetabolite:
    def __init__(self, metabolite_id):
        self.id = metabolite_id
        self.name = metabolite_id
        self.compartment = "c"


class DummyReaction:
    def __init__(self, reaction_id, metabolites):
        self.id = reaction_id
        self.name = reaction_id
        self.metabolites = metabolites
        self.lower_bound = 0
        self.upper_bound = 1000
        self.reversibility = False
        self.gene_reaction_rule = ""
        self.boundary = False


class DummyModel:
    def __init__(self):
        a = DummyMetabolite("a_c")
        b = DummyMetabolite("b_c")
        self.metabolites = [a, b]
        self.reactions = [DummyReaction("R1", {a: -1, b: 1})]


def test_build_bipartite_graph_has_expected_node_types():
    graph = build_bipartite_graph(DummyModel())

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert len(nodes_by_type(graph, METABOLITE)) == 2
    assert len(nodes_by_type(graph, REACTION)) == 1
