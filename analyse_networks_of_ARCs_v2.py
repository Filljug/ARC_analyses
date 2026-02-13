import os
import pandas as pd
import networkx as nx
from itertools import combinations
from math import comb

# Louvain communities (NetworkX >= 2.8 usually)
from networkx.algorithms.community import louvain_communities


# ----------------------------
# Config
# ----------------------------
PAPER_INFRA_LONG = "paper_infra_long.csv"
COMBINED_MAIN = "combined_ARCs_dataset.csv"

OUT_EDGES = "arc_arc_edges_fractional.csv"
OUT_TOP_EDGES = "arc_arc_edges_top25.csv"
OUT_NODE_METRICS = "arc_node_metrics.csv"
OUT_COMMUNITIES = "arc_communities.csv"
OUT_PAPER_ARC_LONG = "paper_arc_long_fractional.csv"

SEED = 123


# ----------------------------
# Helpers
# ----------------------------
def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def standardise_key(s: pd.Series) -> pd.Series:
    """Convert to string, strip, and normalise nulls to empty."""
    return s.fillna("").astype(str).str.strip()


# ----------------------------
# 1) Load infra-long
# ----------------------------
ensure_exists(PAPER_INFRA_LONG)
long = pd.read_csv(PAPER_INFRA_LONG)

required_cols = {"paper_id", "infra_name"}
missing = required_cols - set(long.columns)
if missing:
    raise ValueError(f"{PAPER_INFRA_LONG} is missing columns: {missing}")

long["paper_id"] = standardise_key(long["paper_id"])
long["infra_name"] = long["infra_name"].astype(str).str.strip()

print(f"[OK] Loaded {PAPER_INFRA_LONG}: rows={len(long):,}, unique_papers={long['paper_id'].nunique():,}")
print(f"[OK] Unique ARCs detected: {long['infra_name'].nunique()}")

# ----------------------------
# 2) Build per-paper ARC lists
# ----------------------------
paper_arcs = (
    long.groupby("paper_id")["infra_name"]
    .apply(lambda x: sorted(set(x)))
    .reset_index()
)
paper_arcs["k"] = paper_arcs["infra_name"].apply(len)

n_papers = len(paper_arcs)
n_multi = (paper_arcs["k"] >= 2).sum()
print(f"[OK] Papers with >=2 ARCs (co-attributed): {n_multi:,} / {n_papers:,} ({n_multi/n_papers:.1%})")

# ----------------------------
# 3) Build fractional ARC-ARC edge list
#    Each paper contributes total edge weight = 1
#    weight per pair = 1 / choose(k,2)
# ----------------------------
edges = []
for _, r in paper_arcs.iterrows():
    arcs = r["infra_name"]
    k = r["k"]
    if k < 2:
        continue

    w = 1 / comb(k, 2)
    for a, b in combinations(arcs, 2):
        if a > b:
            a, b = b, a  # consistent ordering
        edges.append((a, b, w))

edge_df = pd.DataFrame(edges, columns=["arc_a", "arc_b", "w"])
edge_df = edge_df.groupby(["arc_a", "arc_b"], as_index=False)["w"].sum()
edge_df = edge_df.sort_values("w", ascending=False)

edge_df.to_csv(OUT_EDGES, index=False)
edge_df.head(25).to_csv(OUT_TOP_EDGES, index=False)

print(f"[OK] Saved weighted edge list -> {OUT_EDGES} (rows={len(edge_df):,})")
print(f"[OK] Saved top 25 edges -> {OUT_TOP_EDGES}")

print("\nTop 10 ARC overlaps (fractional weights):")
print(edge_df.head(10).to_string(index=False))

# ----------------------------
# 4) Build NetworkX graph
# ----------------------------
G = nx.Graph()
for _, r in edge_df.iterrows():
    G.add_edge(r["arc_a"], r["arc_b"], weight=float(r["w"]))

print(f"\n[OK] Graph built: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

# ----------------------------
# 5) Network metrics
# ----------------------------
strength = dict(G.degree(weight="weight"))
betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
pagerank = nx.pagerank(G, weight="weight")

node_metrics = pd.DataFrame({
    "infra_name": list(G.nodes()),
    "strength_weighted_degree": [strength.get(n, 0.0) for n in G.nodes()],
    "betweenness": [betweenness.get(n, 0.0) for n in G.nodes()],
    "pagerank": [pagerank.get(n, 0.0) for n in G.nodes()],
}).sort_values("strength_weighted_degree", ascending=False)

node_metrics.to_csv(OUT_NODE_METRICS, index=False)
print(f"[OK] Saved node metrics -> {OUT_NODE_METRICS}")

# ----------------------------
# 6) Louvain communities
# ----------------------------
communities = louvain_communities(G, weight="weight", seed=SEED)

arc_comm = {}
for i, comm in enumerate(communities):
    for arc in comm:
        arc_comm[arc] = i

comm_df = pd.DataFrame({
    "infra_name": list(G.nodes()),
    "community_id": [arc_comm.get(n, -1) for n in G.nodes()]
}).sort_values(["community_id", "infra_name"])

comm_df.to_csv(OUT_COMMUNITIES, index=False)
print(f"[OK] Saved Louvain communities -> {OUT_COMMUNITIES}")
print("\nCommunities:")
for i, comm in enumerate(communities):
    print(f"  Community {i}: {', '.join(sorted(comm))}")

# ----------------------------
# 7) Create paper*ARC long with fractional weights (1/k)
#    This is for outcome analysis with no double-counting
# ----------------------------
# k per paper (number of unique ARCs)
k_df = long.groupby("paper_id")["infra_name"].nunique().reset_index(name="k")

long2 = long.merge(k_df, on="paper_id", how="left")
long2["w_paper_to_arc"] = 1.0 / long2["k"]

# Attach community label to each ARC row (useful later)
long2 = long2.merge(comm_df, on="infra_name", how="left")

print(f"\n[OK] Paper*ARC attribution table: rows={len(long2):,}")
print("Weight check (should be ~ number of papers):",
      round(long2.groupby("paper_id")["w_paper_to_arc"].sum().sum(), 2))

# ----------------------------
# 8) Merge with main combined dataset
# ----------------------------
ensure_exists(COMBINED_MAIN)
main = pd.read_csv(COMBINED_MAIN)

if "DOI" not in main.columns:
    raise ValueError(f"{COMBINED_MAIN} must include a DOI column")

main["paper_id"] = standardise_key(main["DOI"])

df = long2.merge(main, on="paper_id", how="left", suffixes=("", "_main"))

# Save merged long dataset for outcome modelling
df.to_csv(OUT_PAPER_ARC_LONG, index=False)
print(f"[OK] Saved merged paper*ARC*outcomes dataset -> {OUT_PAPER_ARC_LONG}")

# Quick sanity: missing merges
missing_main = df["Title"].isna().mean() if "Title" in df.columns else df.isna().any(axis=1).mean()
print(f"[INFO] Approx missing merged main fields rate: {missing_main:.1%}")

print("\nDone ?")
