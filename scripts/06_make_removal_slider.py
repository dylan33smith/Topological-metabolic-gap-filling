import json

import networkx as nx

from metabolic_gap_filling.config import FIGURES_DIR, RAW_DATA_DIR
from metabolic_gap_filling.data import load_cobra_model
from metabolic_gap_filling.filters import (
    remove_currency_metabolites,
    select_biomass_reactions,
    select_currency_metabolites,
)
from metabolic_gap_filling.graph import build_bipartite_graph


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = load_cobra_model(RAW_DATA_DIR / "iJO1366.json")
    base_graph = build_bipartite_graph(model)
    no_currency_graph = remove_currency_metabolites(base_graph)

    currency_nodes = select_currency_metabolites(base_graph)
    biomass_nodes = select_biomass_reactions(no_currency_graph)
    actions = [
        removal_action(base_graph, node, "currency metabolite") for node in currency_nodes
    ] + [removal_action(no_currency_graph, node, "biomass reaction") for node in biomass_nodes]

    layout_graph = base_graph.copy()
    pos = nx.spring_layout(layout_graph, seed=33, iterations=45, k=0.08)
    payload = build_payload(base_graph, actions, pos, len(currency_nodes))

    output_path = FIGURES_DIR / "network_removal_slider.html"
    output_path.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Currency removals: {len(currency_nodes)}")
    print(f"Biomass removals: {len(biomass_nodes)}")


def removal_action(graph: nx.Graph, node: str, removal_type: str) -> dict:
    data = graph.nodes[node]
    return {
        "node": node,
        "label": data.get("label", node),
        "node_type": data.get("node_type", ""),
        "degree": graph.degree[node],
        "removal_type": removal_type,
    }


def build_payload(graph: nx.Graph, actions: list[dict], pos: dict, no_currency_step: int) -> dict:
    nodes = []
    node_index = {}
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    for index, node in enumerate(graph.nodes):
        data = graph.nodes[node]
        x, y = pos[node]
        node_index[node] = index
        nodes.append(
            {
                "id": node,
                "label": data.get("label", node),
                "type": data.get("node_type", ""),
                "degree": graph.degree[node],
                "x": normalize(x, min_x, max_x),
                "y": normalize(y, min_y, max_y),
            }
        )

    edges = [[node_index[u], node_index[v]] for u, v in graph.edges]
    snapshots = build_snapshots(graph, actions)
    return {
        "nodes": nodes,
        "edges": edges,
        "actions": actions,
        "snapshots": snapshots,
        "noCurrencyStep": no_currency_step,
        "noCurrencyNoBiomassStep": len(actions),
    }


def build_snapshots(graph: nx.Graph, actions: list[dict]) -> list[dict]:
    working = graph.copy()
    snapshots = [snapshot(working, step=0, removed=None)]
    for step, action in enumerate(actions, start=1):
        if working.has_node(action["node"]):
            working.remove_node(action["node"])
        working.remove_nodes_from(list(nx.isolates(working)))
        snapshots.append(snapshot(working, step=step, removed=action))
    return snapshots


def snapshot(graph: nx.Graph, step: int, removed: dict | None) -> dict:
    components = list(nx.connected_components(graph))
    return {
        "step": step,
        "removed": removed,
        "visible": sorted(graph.nodes),
        "nodeCount": graph.number_of_nodes(),
        "edgeCount": graph.number_of_edges(),
        "componentCount": len(components),
        "largestComponent": max((len(component) for component in components), default=0),
    }


def normalize(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.5
    return 0.05 + 0.90 * ((value - lower) / (upper - lower))


def render_html(payload: dict) -> str:
    data = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Metabolic Network Removal Slider</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #202124;
      background: #f7f7f5;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 700;
    }}
    .panel {{
      background: white;
      border: 1px solid #d9d9d2;
      border-radius: 6px;
      padding: 16px;
      margin-top: 14px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .stat {{
      border-left: 4px solid #4C78A8;
      padding: 6px 8px;
      background: #f5f7fa;
    }}
    .stat strong {{
      display: block;
      font-size: 20px;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 660px;
      background: #ffffff;
      border: 1px solid #d9d9d2;
      border-radius: 4px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    .milestone {{
      margin-top: 8px;
      font-weight: 700;
      color: #B279A2;
    }}
    .tick-labels {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 12px;
      color: #5f6368;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e4e4df;
      padding: 6px 8px;
      text-align: left;
    }}
    th {{
      background: #f1f3f4;
    }}
  </style>
</head>
<body>
<main>
  <h1>Metabolic Network Filtering Sequence</h1>
  <div>Move the slider to remove high-degree currency metabolites first, then biomass reactions.</div>

  <section class="panel">
    <input id="slider" type="range" min="0" max="0" value="0" step="1">
    <div class="tick-labels">
      <span>Base graph</span>
      <span id="currencyTick"></span>
      <span id="finalTick"></span>
    </div>
    <div id="stepLabel"></div>
    <div id="milestone" class="milestone"></div>
    <div class="stats">
      <div class="stat">Nodes<strong id="nodeCount"></strong></div>
      <div class="stat">Edges<strong id="edgeCount"></strong></div>
      <div class="stat">Components<strong id="componentCount"></strong></div>
      <div class="stat">Largest component<strong id="largestComponent"></strong></div>
    </div>
  </section>

  <section class="panel">
    <canvas id="network" width="1120" height="660"></canvas>
  </section>

  <section class="panel">
    <h2>Removal Order</h2>
    <table id="orderTable">
      <thead>
        <tr>
          <th>Step</th>
          <th>Label</th>
          <th>Type</th>
          <th>Degree at selection</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </section>
</main>

<script>
const payload = {data};
const canvas = document.getElementById("network");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("slider");
slider.max = payload.snapshots.length - 1;
document.getElementById("currencyTick").textContent = `no_currency at step ${{payload.noCurrencyStep}}`;
document.getElementById("finalTick").textContent = `no_currency_no_biomass at step ${{payload.noCurrencyNoBiomassStep}}`;

const tbody = document.querySelector("#orderTable tbody");
payload.actions.forEach((action, idx) => {{
  const row = document.createElement("tr");
  row.innerHTML = `<td>${{idx + 1}}</td><td>${{action.label}}</td><td>${{action.node_type}}</td><td>${{action.degree}}</td><td>${{action.removal_type}}</td>`;
  tbody.appendChild(row);
}});

function draw(step) {{
  const snapshot = payload.snapshots[step];
  const visible = new Set(snapshot.visible);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 0.45;
  ctx.strokeStyle = "rgba(120, 120, 120, 0.18)";
  payload.edges.forEach(([a, b]) => {{
    const source = payload.nodes[a];
    const target = payload.nodes[b];
    if (!visible.has(source.id) || !visible.has(target.id)) return;
    ctx.beginPath();
    ctx.moveTo(source.x * canvas.width, source.y * canvas.height);
    ctx.lineTo(target.x * canvas.width, target.y * canvas.height);
    ctx.stroke();
  }});

  payload.nodes.forEach(node => {{
    if (!visible.has(node.id)) return;
    const x = node.x * canvas.width;
    const y = node.y * canvas.height;
    const radius = node.type === "metabolite" ? 2.0 : 1.8;
    ctx.beginPath();
    ctx.fillStyle = node.type === "metabolite" ? "#4C78A8" : "#9D755D";
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }});

  document.getElementById("nodeCount").textContent = snapshot.nodeCount;
  document.getElementById("edgeCount").textContent = snapshot.edgeCount;
  document.getElementById("componentCount").textContent = snapshot.componentCount;
  document.getElementById("largestComponent").textContent = snapshot.largestComponent;

  const removedText = snapshot.removed
    ? `Step ${{step}} removed: ${{snapshot.removed.label}} (${{snapshot.removed.removal_type}}, degree ${{snapshot.removed.degree}})`
    : "Step 0: base graph";
  document.getElementById("stepLabel").textContent = removedText;

  let milestone = "";
  if (step === payload.noCurrencyStep) milestone = "Milestone: this is the no_currency graph.";
  if (step === payload.noCurrencyNoBiomassStep) milestone = "Milestone: this is the no_currency_no_biomass graph.";
  document.getElementById("milestone").textContent = milestone;
}}

slider.addEventListener("input", event => draw(Number(event.target.value)));
draw(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
