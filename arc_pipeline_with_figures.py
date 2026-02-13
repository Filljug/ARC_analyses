import os
import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

import patsy

from statsmodels.duration.hazard_regression import PHReg

from statsmodels.discrete.discrete_model import NegativeBinomial

from itertools import combinations
from math import comb

from networkx.algorithms.community import louvain_communities

# ----------------------------
# Config
# ----------------------------
PAPER_INFRA_LONG = "paper_infra_long.csv"
COMBINED_MAIN = "combined_ARCs_dataset.csv"  # only needed for the merge output

SEED = 123

# Outputs (tables)
OUT_EDGES = "arc_arc_edges_fractional.csv"
OUT_TOP_EDGES_25 = "arc_arc_edges_top25.csv"
OUT_TOP_EDGES_10 = "arc_arc_edges_top10.csv"
OUT_NODE_METRICS = "arc_node_metrics.csv"
OUT_COMMUNITIES = "arc_communities.csv"
OUT_CENTRALITY_TOP10 = "centrality_top10_tables.csv"
OUT_PAPER_ARC_LONG = "paper_arc_long_fractional.csv"

# Outputs (figures)
FIG_NETWORK = "fig_arc_network_map.png"
FIG_NETWORK_PDF = "fig_arc_network_map.pdf"
FIG_SCATTER = "fig_strength_vs_betweenness.png"
FIG_SCATTER_PDF = "fig_strength_vs_betweenness.pdf"
FIG_COMMUNITY = "fig_community_strength.png"
FIG_COMMUNITY_PDF = "fig_community_strength.pdf"

# Plot sizing/scaling controls (tweakable)
NODE_SIZE_MIN = 600
NODE_SIZE_MAX = 3600
EDGE_WIDTH_MIN = 0.5
EDGE_WIDTH_MAX = 6.0


# ----------------------------
# Helpers
# ----------------------------
def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def standardise_key(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def minmax_scale(values, out_min, out_max):
    """Scale a list/Series to [out_min, out_max]."""
    v = pd.Series(values, dtype=float)
    if v.nunique() <= 1:
        return [0.5 * (out_min + out_max)] * len(v)
    v_scaled = (v - v.min()) / (v.max() - v.min())
    return (out_min + v_scaled * (out_max - out_min)).tolist()


def save_fig(path_png, path_pdf=None):
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    if path_pdf:
        plt.savefig(path_pdf)
    plt.close()


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
#    Each paper contributes total edge weight = 1 across its pairs
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
edge_df.head(25).to_csv(OUT_TOP_EDGES_25, index=False)
edge_df.head(10).to_csv(OUT_TOP_EDGES_10, index=False)

print(f"[OK] Saved weighted edge list -> {OUT_EDGES} (rows={len(edge_df):,})")
print(f"[OK] Saved top 25 edges -> {OUT_TOP_EDGES_25}")
print(f"[OK] Saved top 10 edges -> {OUT_TOP_EDGES_10}")

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
# 7) Centrality ranking tables (Top 10)
# ----------------------------
top10_strength = node_metrics.sort_values("strength_weighted_degree", ascending=False).head(10).copy()
top10_strength["rank_by"] = "strength"

top10_betw = node_metrics.sort_values("betweenness", ascending=False).head(10).copy()
top10_betw["rank_by"] = "betweenness"

centrality_top10 = pd.concat([top10_strength, top10_betw], ignore_index=True)
centrality_top10 = centrality_top10.merge(comm_df, on="infra_name", how="left")

centrality_top10.to_csv(OUT_CENTRALITY_TOP10, index=False)
print(f"[OK] Saved centrality top-10 tables -> {OUT_CENTRALITY_TOP10}")

# ----------------------------
# 8) Create paper*ARC long with fractional weights (1/k)
# ----------------------------
k_df = long.groupby("paper_id")["infra_name"].nunique().reset_index(name="k")
long2 = long.merge(k_df, on="paper_id", how="left")
long2["w_paper_to_arc"] = 1.0 / long2["k"]
long2 = long2.merge(comm_df, on="infra_name", how="left")

# ----------------------------
# 9) Merge with main combined dataset (optional but recommended)
# ----------------------------
if os.path.exists(COMBINED_MAIN):
    main = pd.read_csv(COMBINED_MAIN)
    if "DOI" not in main.columns:
        raise ValueError(f"{COMBINED_MAIN} must include a DOI column")

    main["paper_id"] = standardise_key(main["DOI"])
    df = long2.merge(main, on="paper_id", how="left", suffixes=("", "_main"))
    df.to_csv(OUT_PAPER_ARC_LONG, index=False)
    print(f"[OK] Saved merged paper*ARC*outcomes dataset -> {OUT_PAPER_ARC_LONG}")
else:
    print(f"[WARN] {COMBINED_MAIN} not found; skipping merge output.")

# ============================================================
# FIGURE 1: ARC NETWORK MAP (paper-ready)
# node size = strength
# edge width = weight
# node colour = Louvain community
# ============================================================
# Layout
pos = nx.spring_layout(G, seed=SEED, weight="weight", k=0.9)

# Node sizes (scaled)
node_strengths = [strength.get(n, 0.0) for n in G.nodes()]
node_sizes = minmax_scale(node_strengths, NODE_SIZE_MIN, NODE_SIZE_MAX)

# Edge widths (scaled)
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
edge_widths = minmax_scale(edge_weights, EDGE_WIDTH_MIN, EDGE_WIDTH_MAX)

# Node colours by community
node_communities = [arc_comm.get(n, -1) for n in G.nodes()]
cmap = plt.cm.tab20  # plenty of distinct colours for <= 20 communities

plt.figure(figsize=(12, 10))
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_communities,
    cmap=cmap,
    alpha=0.9
)

# Labels
nx.draw_networkx_labels(G, pos, font_size=9)

plt.title("NIHR ARC/CLAHRC Collaboration Network (ARC-ARC co-attribution)\n"
          "Node size = weighted strength; Edge width = fractional overlap; Colour = Louvain community")
plt.axis("off")
save_fig(FIG_NETWORK, FIG_NETWORK_PDF)
print(f"[OK] Saved network figure -> {FIG_NETWORK} (+ PDF)")

# ============================================================
# FIGURE 2: Strength vs Betweenness (volume vs brokerage)
# ============================================================
plot_df = node_metrics.merge(comm_df, on="infra_name", how="left")

plt.figure(figsize=(10, 7))
plt.scatter(plot_df["strength_weighted_degree"], plot_df["betweenness"])

# Label the top 5 by betweenness (often broker ARCs)
top_broker = plot_df.sort_values("betweenness", ascending=False).head(5)
for _, r in top_broker.iterrows():
    plt.annotate(r["infra_name"],
                 (r["strength_weighted_degree"], r["betweenness"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=9)

plt.xlabel("Strength (weighted degree)")
plt.ylabel("Betweenness centrality")
plt.title("ARC roles: output volume vs brokerage position")
save_fig(FIG_SCATTER, FIG_SCATTER_PDF)
print(f"[OK] Saved scatter figure -> {FIG_SCATTER} (+ PDF)")

# ============================================================
# FIGURE 3: Community composition (total strength by community)
# ============================================================
comm_strength = plot_df.groupby("community_id", as_index=False)["strength_weighted_degree"].sum()
comm_strength = comm_strength.sort_values("strength_weighted_degree", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(comm_strength["community_id"].astype(str), comm_strength["strength_weighted_degree"])
plt.xlabel("Louvain community")
plt.ylabel("Total strength (sum of node strengths)")
plt.title("Collaboration blocs: community strength (sum of weighted degrees)")
save_fig(FIG_COMMUNITY, FIG_COMMUNITY_PDF)
print(f"[OK] Saved community bar figure -> {FIG_COMMUNITY} (+ PDF)")

# ============================================================
# ARC Collaboration Summary: collaborative vs non-collaborative
# ============================================================

OUT_ARC_COLLAB_SUMMARY = "arc_collaboration_summary.csv"
OUT_ARC_TOP_PARTNERS = "arc_collaboration_top_partners.csv"

# 1) Ensure we have k and weights per paper*ARC
# long2 already contains: paper_id, infra_name, k, w_paper_to_arc
# If your script uses different names, adjust accordingly.

# Mark whether each paper is multi-ARC (collaborative) or single-ARC
long2["is_collaborative_paper"] = (long2["k"] >= 2).astype(int)

# 2) ARC-level summary using fractional output weights
arc_summary = (
    long2.groupby("infra_name", as_index=False)
    .agg(
        # Total fractional volume of output for this ARC
        total_output_volume=("w_paper_to_arc", "sum"),

        # Fractional volume from collaborative papers only
        collaborative_output_volume=("w_paper_to_arc",
                                    lambda s: s[long2.loc[s.index, "is_collaborative_paper"] == 1].sum()),

        # Fractional volume from non-collaborative (single-ARC) papers only
        noncollab_output_volume=("w_paper_to_arc",
                                 lambda s: s[long2.loc[s.index, "is_collaborative_paper"] == 0].sum()),

        # How many distinct papers mention this ARC (raw, not fractional)
        n_papers_raw=("paper_id", "nunique"),

        # Average number of ARCs per paper for this ARC's papers
        mean_arcs_per_paper=("k", "mean"),

        # % of this ARC's output that is collaborative (fractional basis)
        # (computed after aggregation)
    )
)

arc_summary["collab_share_of_output"] = (
    arc_summary["collaborative_output_volume"] / arc_summary["total_output_volume"]
)

# Nice formatting: sort by total volume
arc_summary = arc_summary.sort_values("total_output_volume", ascending=False)

# Save
arc_summary.to_csv(OUT_ARC_COLLAB_SUMMARY, index=False)
print(f"[OK] Saved ARC collaboration summary -> {OUT_ARC_COLLAB_SUMMARY}")

print("\nTop 10 ARCs by total output volume (fractional):")
print(arc_summary.head(10)[
    ["infra_name", "total_output_volume", "collaborative_output_volume",
     "noncollab_output_volume", "collab_share_of_output", "mean_arcs_per_paper"]
].to_string(index=False))

# ------------------------------------------------------------
# OPTIONAL: Top partners per ARC (uses the weighted edge list)
# ------------------------------------------------------------
# Requires: edge_df with columns arc_a, arc_b, w
# Produces top 5 partners per ARC based on overlap weight

partner_rows = []
for arc in pd.unique(edge_df[["arc_a", "arc_b"]].values.ravel("K")):
    # edges where arc is involved
    sub = edge_df[(edge_df["arc_a"] == arc) | (edge_df["arc_b"] == arc)].copy()
    if sub.empty:
        continue
    sub["partner"] = sub.apply(lambda r: r["arc_b"] if r["arc_a"] == arc else r["arc_a"], axis=1)
    sub = sub.sort_values("w", ascending=False).head(5)

    for _, r in sub.iterrows():
        partner_rows.append({
            "infra_name": arc,
            "top_partner": r["partner"],
            "overlap_weight": r["w"]
        })

partners_df = pd.DataFrame(partner_rows)
partners_df.to_csv(OUT_ARC_TOP_PARTNERS, index=False)
print(f"[OK] Saved ARC top partners -> {OUT_ARC_TOP_PARTNERS}")

# ============================================================
# FIGURE: Stacked bar chart of ARC output (collab vs non-collab)
# ============================================================

FIG_ARC_STACKED = "fig_arc_output_collab_stacked.png"
FIG_ARC_STACKED_PDF = "fig_arc_output_collab_stacked.pdf"

# Order ARCs by total fractional volume (already sorted)
plot_arc = arc_summary.copy()

# Create stacked bar chart
plt.figure(figsize=(12, 7))

x = plot_arc["infra_name"]
noncollab = plot_arc["noncollab_output_volume"]
collab = plot_arc["collaborative_output_volume"]

plt.bar(x, noncollab, label="Non-collaborative (single-ARC papers)")
plt.bar(x, collab, bottom=noncollab, label="Collaborative (multi-ARC papers)")

plt.xticks(rotation=45, ha="right")
plt.xlabel("ARC")
plt.ylabel("Fractional output volume (sum of 1/k weights)")
plt.title("ARC output volume: collaborative vs non-collaborative papers\n(Fractional counting; each paper contributes total weight = 1)")
plt.legend()

plt.tight_layout()
plt.savefig(FIG_ARC_STACKED, dpi=300)
plt.savefig(FIG_ARC_STACKED_PDF)
plt.close()

print(f"[OK] Saved stacked bar figure -> {FIG_ARC_STACKED} (+ PDF)")

# ============================================================
# OUTCOME MODULES: ARC identity + network position -> outcomes
# ============================================================

# ----------------------------
# Output filenames
# ----------------------------
OUT_ARC_OUTCOMES = "arc_outcomes_summary.csv"
OUT_COMM_OUTCOMES = "community_outcomes_summary.csv"
OUT_MULTI_SINGLE = "multiARC_vs_singleARC_summary.csv"

FIG_CENTRALITY_POLICY = "fig_centrality_vs_policy_rate.png"
FIG_CENTRALITY_POLICY_PDF = "fig_centrality_vs_policy_rate.pdf"

FIG_CENTRALITY_CITES = "fig_centrality_vs_citation_rate.png"
FIG_CENTRALITY_CITES_PDF = "fig_centrality_vs_citation_rate.pdf"

FIG_MULTI_POLICY = "fig_multiARC_vs_singleARC_policy_rate.png"
FIG_MULTI_POLICY_PDF = "fig_multiARC_vs_singleARC_policy_rate.pdf"

FIG_MULTI_CITES = "fig_multiARC_vs_singleARC_citations_box.png"
FIG_MULTI_CITES_PDF = "fig_multiARC_vs_singleARC_citations_box.pdf"

FIG_TREND_COLLAB = "fig_trend_multiARC_share.png"
FIG_TREND_COLLAB_PDF = "fig_trend_multiARC_share.pdf"

FIG_TREND_POLICY = "fig_trend_policy_rate.png"
FIG_TREND_POLICY_PDF = "fig_trend_policy_rate.pdf"

FIG_TREND_OA = "fig_trend_open_access_share.png"
FIG_TREND_OA_PDF = "fig_trend_open_access_share.pdf"


# ----------------------------
# Helpers
# ----------------------------
def to_numeric_clean(x):
    """Convert to numeric, treating '.' and empty strings as missing."""
    return pd.to_numeric(x.replace(".", np.nan) if isinstance(x, pd.Series) else x, errors="coerce")


import re
import pandas as pd

def parse_partial_date(s, anchor_day_month=15, anchor_month_year=7, anchor_day_year=1):
    """
    Parses:
      - YYYY-MM-DD -> exact
      - YYYY-MM    -> day anchored (default 15)
      - YYYY       -> month/day anchored (default Jul 1)
    Anything else -> NaT
    """
    s = str(s).strip()
    if s in ["", "nan", "None", "."]:
        return pd.NaT

    # YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return pd.to_datetime(s, errors="coerce")

    # YYYY-MM
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return pd.to_datetime(f"{s}-{anchor_day_month:02d}", errors="coerce")

    # YYYY
    if re.fullmatch(r"\d{4}", s):
        return pd.to_datetime(f"{s}-{anchor_month_year:02d}-{anchor_day_year:02d}", errors="coerce")

    # fallback: try parse (rare; won't harm)
    return pd.to_datetime(s, errors="coerce")


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    if mask.sum() == 0:
        return np.nan
    return np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])


def weighted_rate(binary, weights):
    binary = np.asarray(binary, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(binary) & np.isfinite(weights)
    if mask.sum() == 0:
        return np.nan
    return np.sum(binary[mask] * weights[mask]) / np.sum(weights[mask])


def safe_savefig(png_path, pdf_path=None):
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    if pdf_path:
        plt.savefig(pdf_path)
    plt.close()


# ----------------------------
# 1) Prepare outcomes + dates
# ----------------------------
required_main_cols = [
    "Title", "PubYear", "Publication date (online)", "Publication date (print)",
    "Open Access", "Times cited", "FCR", "Altmetric",
    "Fields of Research (ANZSRC 2020)", "Policy mentions"
]
missing_cols = [c for c in required_main_cols if c not in df.columns]
if missing_cols:
    print("[WARN] Missing some combined dataset columns in merged df:", missing_cols)

# Parse prioritised publication date:
df["date_online"] = df["Publication date (online)"].apply(parse_partial_date)
df["date_print"]  = df["Publication date (print)"].apply(parse_partial_date)
df["date_year"]   = df["PubYear"].apply(parse_partial_date)

online_str = df["Publication date (online)"].astype(str).str.strip()
is_online_year_only = online_str.str.fullmatch(r"\d{4}")

df["pub_date_final"] = df["date_online"]
df.loc[is_online_year_only & df["date_print"].notna(), "pub_date_final"] = df.loc[is_online_year_only & df["date_print"].notna(), "date_print"]

# then still fall back where needed
df.loc[df["pub_date_final"].isna(), "pub_date_final"] = df.loc[df["pub_date_final"].isna(), "date_print"]
df.loc[df["pub_date_final"].isna(), "pub_date_final"] = df.loc[df["pub_date_final"].isna(), "date_year"]

df["pub_year_final"] = df["pub_date_final"].dt.year

# Clean numeric outcomes
if "Times cited" in df.columns:
    df["times_cited_num"] = to_numeric_clean(df["Times cited"])
else:
    df["times_cited_num"] = np.nan

if "Policy mentions" in df.columns:
    df["policy_mentions_num"] = to_numeric_clean(df["Policy mentions"])
else:
    df["policy_mentions_num"] = np.nan

if "Altmetric" in df.columns:
    df["altmetric_num"] = to_numeric_clean(df["Altmetric"])
else:
    df["altmetric_num"] = np.nan

if "FCR" in df.columns:
    df["fcr_num"] = to_numeric_clean(df["FCR"])
else:
    df["fcr_num"] = np.nan

df["policy_any"] = np.where(df["policy_mentions_num"].fillna(0) >= 1, 1, 0)

# Age-normalised citations (important fairness)
today = pd.Timestamp.today().normalize()
df["years_since_pub"] = (today - df["pub_date_final"]).dt.days / 365.25
df.loc[df["years_since_pub"] <= 0, "years_since_pub"] = np.nan
df["citations_per_year"] = df["times_cited_num"] / df["years_since_pub"]

# Policy mentions per year (exposure-normalised)
df["policy_mentions_per_year"] = df["policy_mentions_num"] / df["years_since_pub"]

# Optional stability rule: avoid extreme rates for very recent papers
df.loc[df["years_since_pub"] < 0.5, "policy_mentions_per_year"] = np.nan


# ----------------------------
# 2) ARC outcomes dashboard (fractional weights)
# ----------------------------
# One row per paper×ARC, weighted by w_paper_to_arc (1/k)
# This avoids double-counting multi-ARC papers
if "w_paper_to_arc" not in df.columns:
    raise ValueError("Expected df to contain w_paper_to_arc. Ensure you created 1/k weights earlier.")

arc_out_rows = []
for arc, sub in df.groupby("infra_name"):
    w = sub["w_paper_to_arc"].astype(float)

    arc_out_rows.append({
        "infra_name": arc,
        "community_id": int(sub["community_id"].dropna().iloc[0]) if "community_id" in sub.columns and sub["community_id"].notna().any() else np.nan,

        # Output volume (fractional)
        "output_volume_fractional": float(w.sum()),
        "papers_raw": int(sub["paper_id"].nunique()),

        # Collaboration intensity (from k)
        "mean_ARCs_per_paper": float(sub["k"].mean()) if "k" in sub.columns else np.nan,
        "share_multiARC_papers_raw": float((sub.drop_duplicates("paper_id")["k"] >= 2).mean()) if "k" in sub.columns else np.nan,

        # Policy
        "policy_rate_weighted": weighted_rate(sub["policy_any"], w),
        "policy_mentions_mean_weighted": weighted_mean(sub["policy_mentions_num"], w),

        # Citations
        "citations_mean_weighted": weighted_mean(sub["times_cited_num"], w),
        "citations_per_year_mean_weighted": weighted_mean(sub["citations_per_year"], w),

        # Other outcomes
        "altmetric_mean_weighted": weighted_mean(sub["altmetric_num"], w),
        "fcr_mean_weighted": weighted_mean(sub["fcr_num"], w),

        # Open access share
        "open_access_share_weighted": weighted_rate(sub["Open Access"].astype(str).str.strip().ne("").astype(int) if "Open Access" in sub.columns else np.nan, w)
    })

arc_out = pd.DataFrame(arc_out_rows)
arc_out = arc_out.sort_values("output_volume_fractional", ascending=False)

arc_out.to_csv(OUT_ARC_OUTCOMES, index=False)
print(f"[OK] Saved ARC outcomes dashboard -> {OUT_ARC_OUTCOMES}")

# Quick headline preview
print("\nTop 8 ARCs by fractional output volume:")
print(arc_out.head(8)[[
    "infra_name", "output_volume_fractional",
    "policy_rate_weighted", "citations_per_year_mean_weighted",
    "mean_ARCs_per_paper"
]].to_string(index=False))


# ----------------------------
# 3) Community outcome profiles (Louvain blocs)
# ----------------------------
if "community_id" in df.columns:
    comm_rows = []
    for comm, sub in df.groupby("community_id"):
        w = sub["w_paper_to_arc"].astype(float)
        comm_rows.append({
            "community_id": int(comm),
            "n_arcs_in_comm": int(sub["infra_name"].nunique()),
            "output_volume_fractional": float(w.sum()),
            "policy_rate_weighted": weighted_rate(sub["policy_any"], w),
            "policy_mentions_mean_weighted": weighted_mean(sub["policy_mentions_num"], w),
            "citations_mean_weighted": weighted_mean(sub["times_cited_num"], w),
            "citations_per_year_mean_weighted": weighted_mean(sub["citations_per_year"], w),
            "altmetric_mean_weighted": weighted_mean(sub["altmetric_num"], w),
            "fcr_mean_weighted": weighted_mean(sub["fcr_num"], w),
        })
    comm_out = pd.DataFrame(comm_rows).sort_values("output_volume_fractional", ascending=False)
    comm_out.to_csv(OUT_COMM_OUTCOMES, index=False)
    print(f"[OK] Saved community outcome profiles -> {OUT_COMM_OUTCOMES}")
else:
    print("[WARN] community_id not present; skipping community outcome profiles.")
    comm_out = None


# ----------------------------
# 4) Centrality vs outcomes (ARC-level)
# ----------------------------
# Join node_metrics (centrality) with ARC outcomes dashboard
# node_metrics should already exist from your earlier code
arc_central = arc_out.merge(node_metrics, on="infra_name", how="left")

# Spearman correlations (robust for small N=15 ARCs)
corr_fields = [
    ("strength_weighted_degree", "policy_rate_weighted"),
    ("betweenness", "policy_rate_weighted"),
    ("strength_weighted_degree", "citations_per_year_mean_weighted"),
    ("betweenness", "citations_per_year_mean_weighted"),
]
print("\n[INFO] Spearman correlations (ARC-level):")
for x, y in corr_fields:
    tmp = arc_central[[x, y]].dropna()
    if len(tmp) >= 3:
        rho = tmp[x].corr(tmp[y], method="spearman")
        print(f"  {x} vs {y}: rho={rho:.3f} (n={len(tmp)})")

# Plot: Strength vs policy rate
plt.figure(figsize=(10, 7))
plt.scatter(arc_central["strength_weighted_degree"], arc_central["policy_rate_weighted"])
for _, r in arc_central.sort_values("policy_rate_weighted", ascending=False).head(5).iterrows():
    plt.annotate(r["infra_name"], (r["strength_weighted_degree"], r["policy_rate_weighted"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=9)
plt.xlabel("Strength (weighted degree): collaboration volume")
plt.ylabel("Policy visibility rate (weighted % with ≥1 policy mention)")
plt.title("Network position and policy visibility by ARC")
safe_savefig(FIG_CENTRALITY_POLICY, FIG_CENTRALITY_POLICY_PDF)
print(f"[OK] Saved centrality-policy figure -> {FIG_CENTRALITY_POLICY} (+ PDF)")

# Plot: Strength vs citation rate
plt.figure(figsize=(10, 7))
plt.scatter(arc_central["strength_weighted_degree"], arc_central["citations_per_year_mean_weighted"])
for _, r in arc_central.sort_values("citations_per_year_mean_weighted", ascending=False).head(5).iterrows():
    plt.annotate(r["infra_name"], (r["strength_weighted_degree"], r["citations_per_year_mean_weighted"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=9)
plt.xlabel("Strength (weighted degree): collaboration volume")
plt.ylabel("Citations per year since publication (weighted mean)")
plt.title("Network position and citation rate by ARC (age-normalised)")
safe_savefig(FIG_CENTRALITY_CITES, FIG_CENTRALITY_CITES_PDF)
print(f"[OK] Saved centrality-citations figure -> {FIG_CENTRALITY_CITES} (+ PDF)")


# ----------------------------
# 5) Multi-ARC vs single-ARC outcomes (paper-level, no duplication)
# ----------------------------
# Create unique paper-level table: paper_id -> k and outcomes
paper_k = long.groupby("paper_id")["infra_name"].nunique().reset_index(name="k_arc")

# Start from main combined dataset (unique DOI)
paper_main = main.copy()
paper_main["paper_id"] = standardise_key(paper_main["DOI"])

paper_level = paper_main.merge(paper_k, on="paper_id", how="left")
paper_level["k_arc"] = paper_level["k_arc"].fillna(0).astype(int)
paper_level["is_multiARC"] = (paper_level["k_arc"] >= 2).astype(int)
paper_level["is_singleARC"] = (paper_level["k_arc"] == 1).astype(int)

# Clean outcomes at paper-level
paper_level["times_cited_num"] = to_numeric_clean(paper_level["Times cited"])
paper_level["policy_mentions_num"] = to_numeric_clean(paper_level["Policy mentions"])
paper_level["policy_any"] = np.where(paper_level["policy_mentions_num"].fillna(0) >= 1, 1, 0)

# Publication date at paper-level
paper_level["date_online"] = paper_level["Publication date (online)"].apply(parse_partial_date)
paper_level["date_print"] = paper_level["Publication date (print)"].apply(parse_partial_date)
paper_level["date_year"] = paper_level["PubYear"].apply(parse_partial_date)
paper_level["pub_date_final"] = paper_level["date_online"]
paper_level.loc[paper_level["pub_date_final"].isna(), "pub_date_final"] = paper_level.loc[paper_level["pub_date_final"].isna(), "date_print"]
paper_level.loc[paper_level["pub_date_final"].isna(), "pub_date_final"] = paper_level.loc[paper_level["pub_date_final"].isna(), "date_year"]

paper_level["years_since_pub"] = (today - paper_level["pub_date_final"]).dt.days / 365.25
paper_level.loc[paper_level["years_since_pub"] <= 0, "years_since_pub"] = np.nan
paper_level["citations_per_year"] = paper_level["times_cited_num"] / paper_level["years_since_pub"]

# ============================================================
# ALT METRICS: time from publication -> first policy mention
#   - Links by DOI
#   - Uses Altmetric "Mention Date" (dd/mm/yyyy hh:mm), time ignored
#   - Keeps only FIRST policy mention per DOI
# ============================================================

ALTMETRIC_POLICY_CSV = "Altmetric_ARCs_policy_dates.csv"  # <-- change to your filename

def clean_doi_series(s: pd.Series) -> pd.Series:
    """
    Standardise DOIs to improve matching:
      - strip whitespace
      - lowercase
      - remove common URL prefixes
    """
    s = s.fillna("").astype(str).str.strip().str.lower()
    s = s.str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    s = s.str.replace(r"^doi:\s*", "", regex=True)
    return s

ensure_exists(ALTMETRIC_POLICY_CSV)
alt = pd.read_csv(ALTMETRIC_POLICY_CSV)

# --- required columns in the Altmetric file
req_alt_cols = {"DOI", "Mention Date"}
missing_alt = req_alt_cols - set(alt.columns)
if missing_alt:
    raise ValueError(f"{ALTMETRIC_POLICY_CSV} is missing columns: {missing_alt}")

# --- parse Mention Date (dd/mm/yyyy hh:mm); ignore time
alt["doi_clean"] = clean_doi_series(alt["DOI"])
alt["mention_date"] = pd.to_datetime(
    alt["Mention Date"].astype(str).str.slice(0, 10),  # keep dd/mm/yyyy
    format="%d/%m/%Y",
    errors="coerce"
)

# --- first policy mention per DOI
first_mention = (
    alt.dropna(subset=["doi_clean", "mention_date"])
       .groupby("doi_clean", as_index=False)["mention_date"]
       .min()
       .rename(columns={"mention_date": "first_policy_mention_date"})
)

print(f"[OK] Altmetric policy file loaded: rows={len(alt):,}, DOIs with ≥1 parsed mention={len(first_mention):,}")

# --- attach to df (paper×ARC long) if present
if "paper_id" in df.columns:
    df["doi_clean"] = clean_doi_series(df["paper_id"])  # paper_id is DOI in your pipeline
    df = df.merge(first_mention, on="doi_clean", how="left")

    # days from publication -> first policy mention
    df["days_to_first_policy_mention"] = (
        df["first_policy_mention_date"] - df["pub_date_final"]
    ).dt.days

    # optional sanity rule: negative lags are likely data issues
    df.loc[df["days_to_first_policy_mention"] < 0, "days_to_first_policy_mention"] = np.nan

# --- attach to paper_level (unique DOI table) for modelling/summaries
paper_level["doi_clean"] = clean_doi_series(paper_level["DOI"])
paper_level = paper_level.merge(first_mention, on="doi_clean", how="left")

paper_level["days_to_first_policy_mention"] = (
    paper_level["first_policy_mention_date"] - paper_level["pub_date_final"]
).dt.days
paper_level.loc[paper_level["days_to_first_policy_mention"] < 0, "days_to_first_policy_mention"] = np.nan

# Optional quick checks
n_with_policy_date = paper_level["first_policy_mention_date"].notna().sum()
print(f"[INFO] Papers with a first policy mention date matched: {n_with_policy_date:,}")
print("[INFO] days_to_first_policy_mention summary (non-missing):")
print(paper_level["days_to_first_policy_mention"].dropna().describe())

# ============================================================
# TIME TO FIRST POLICY MENTION ~ multi-ARC involvement
#   Main: Cox PH (handles censoring for no-mention papers)
#   Robustness: OLS on log(1+days) among mentioned-only papers
# ============================================================


OUT_TTFP_COX = "model_time_to_first_policy_mention_cox.txt"
OUT_TTFP_OLS = "model_time_to_first_policy_mention_ols.txt"

# --- guardrails: require the altmetric-derived columns
req_cols = {"first_policy_mention_date", "days_to_first_policy_mention", "pub_date_final", "k_arc", "Open Access"}
missing = req_cols - set(paper_level.columns)
if missing:
    raise ValueError(
        f"Missing required columns for time-to-first-policy modelling: {missing}. "
        "Have you merged the Altmetric policy mention dates into paper_level yet?"
    )

ttfp = paper_level.copy()

# Multi-ARC indicator (matches your other models)
ttfp["is_multiARC"] = (ttfp["k_arc"] >= 2).astype(int)

# OA binary (reuse your logic, simplified: closed=0 else=1)
oa = ttfp["Open Access"].astype(str).str.strip().str.lower()
ttfp["OpenAccess_bin"] = np.where(oa.str.contains(r"\bclosed\b", na=False), 0, 1)

# FoR category (if present)
if "Fields of Research (ANZSRC 2020)" in ttfp.columns:
    ttfp["FoR"] = ttfp["Fields of Research (ANZSRC 2020)"].astype(str).fillna("unknown")
else:
    ttfp["FoR"] = "unknown"

# Publication year + centered year
if "pub_year_final" not in ttfp.columns:
    ttfp["pub_year_final"] = ttfp["pub_date_final"].dt.year
ttfp["pub_year_final"] = pd.to_numeric(ttfp["pub_year_final"], errors="coerce")
ttfp = ttfp.dropna(subset=["pub_year_final", "pub_date_final"])
ttfp["pub_year_c"] = ttfp["pub_year_final"] - ttfp["pub_year_final"].mean()

# Event indicator + time variable with censoring at `today` (already used elsewhere in your script)
ttfp["event"] = ttfp["first_policy_mention_date"].notna().astype(int)

# time-to-event for those with mention dates
ttfp["t_event"] = pd.to_numeric(ttfp["days_to_first_policy_mention"], errors="coerce")

# censoring time for those without a mention date: time from pub -> today
ttfp["t_censor"] = (today - ttfp["pub_date_final"]).dt.days

# combined analysis time
ttfp["t"] = np.where(ttfp["event"] == 1, ttfp["t_event"], ttfp["t_censor"])
ttfp["t"] = pd.to_numeric(ttfp["t"], errors="coerce")

# Clean up impossible values
ttfp.loc[ttfp["t"] <= 0, "t"] = np.nan
ttfp = ttfp.dropna(subset=["t", "event", "is_multiARC", "OpenAccess_bin", "pub_year_c"])

print(f"[INFO] TTFP modelling rows={len(ttfp):,} (events={int(ttfp['event'].sum()):,}, censored={int((1-ttfp['event']).sum()):,})")

# ----------------------------
# 1) Cox proportional hazards (PHReg)
# Interpretation: HR>1 means *faster* first mention (shorter time)
# ----------------------------
# ----------------------------
# Cox proportional hazards (PHReg) -- robust + non-singular
# ----------------------------
import numpy as np
import pandas as pd
import patsy
from statsmodels.duration.hazard_regression import PHReg

def drop_constant_cols_df(X):
    keep = []
    for c in X.columns:
        if X[c].dropna().nunique() <= 1:
            continue
        keep.append(c)
    return X[keep]

def drop_exact_collinear_cols_df(X, tol=1e-12):
    M = X.to_numpy(dtype=float)
    kept_idx, rank = [], 0
    for j in range(M.shape[1]):
        trial = kept_idx + [j]
        r = np.linalg.matrix_rank(M[:, trial], tol=tol)
        if r > rank:
            kept_idx.append(j)
            rank = r
    return X.iloc[:, kept_idx]

# ---- 1) Ensure types (do this once, early)
ttfp["FoR"] = ttfp["FoR"].astype(str).fillna("unknown")
ttfp["OpenAccess_bin"] = pd.to_numeric(ttfp["OpenAccess_bin"], errors="coerce")
ttfp["t"] = pd.to_numeric(ttfp["t"], errors="coerce")
ttfp["event"] = pd.to_numeric(ttfp["event"], errors="coerce")
ttfp["k_arc"] = pd.to_numeric(ttfp["k_arc"], errors="coerce")
ttfp["pub_year_final"] = pd.to_numeric(ttfp["pub_year_final"], errors="coerce")

# ---- 2) Filter to plausible years first
ttfp = ttfp[ttfp["pub_year_final"].between(2010, 2025, inclusive="both")].copy()

# ---- 3) Drop rows missing key survival fields/predictors BEFORE banding
ttfp = ttfp.dropna(subset=["t", "event", "k_arc", "OpenAccess_bin", "pub_year_final"]).copy()

# Optional: keep only non-negative time
ttfp = ttfp[ttfp["t"] >= 0].copy()

# ---- 4) Create 3-year publication bands (int-safe)
ttfp["pub_year_band3"] = ((ttfp["pub_year_final"].astype(int) // 3) * 3).astype(int)

# ---- 5) Drop year bands that are too sparse / too few events
band_stats = ttfp.groupby("pub_year_band3").agg(
    n=("event", "size"),
    events=("event", "sum")
).reset_index()

good_bands = band_stats[
    (band_stats["n"] >= 30) &
    (band_stats["events"] >= 10) &
    ((band_stats["n"] - band_stats["events"]) >= 10)
]["pub_year_band3"]

ttfp = ttfp[ttfp["pub_year_band3"].isin(good_bands)].copy()

# ---- 6) Collapse collaboration to 1 vs 2+ (avoid sparse bins)
ttfp["k_arc_bin"] = np.where(ttfp["k_arc"] >= 2, "2plus", "1")
ttfp["k_arc_bin"] = pd.Categorical(ttfp["k_arc_bin"], categories=["1", "2plus"])

# ---- 7) Build design matrix (NO intercept)
X = patsy.dmatrix(
    '0 + C(k_arc_bin, Treatment(reference="1")) + OpenAccess_bin + C(pub_year_band3)',
    data=ttfp,
    return_type="dataframe",
    NA_action="drop"
)

ttfp2 = ttfp.loc[X.index].copy()

# ---- 8) Diagnostics
print(f"[INFO] Cox rows after Patsy drop: {len(ttfp2):,}")
print("[INFO] Events:", ttfp2["event"].value_counts(dropna=False).to_dict())
print("[INFO] OpenAccess_bin:", ttfp2["OpenAccess_bin"].value_counts(dropna=False).to_dict())
print("[INFO] k_arc_bin:", ttfp2["k_arc_bin"].value_counts(dropna=False).to_dict())
print("[INFO] pub_year_band3:", ttfp2["pub_year_band3"].value_counts(dropna=False).head(10).to_dict())

# ---- 9) Drop constant/collinear cols in X
X = drop_constant_cols_df(X)
X = drop_exact_collinear_cols_df(X)
print("[INFO] Cox design matrix shape after dropping constant/collinear:", X.shape)

# ---- 10) Fit Cox
cox_fit = PHReg(
    endog=ttfp2["t"].astype(float),
    exog=X.astype(float),
    status=ttfp2["event"].astype(int)
).fit()

with open(OUT_TTFP_COX, "w", encoding="utf-8") as f:
    f.write("MODEL=Cox proportional hazards (PHReg)\n")
    f.write("Outcome=time to first policy mention (days), censored at today when no mention\n")
    f.write("Covariates= k_arc_bin (2plus vs 1) + OpenAccess_bin + pub_year_band3 (no intercept)\n\n")
    f.write(cox_fit.summary().as_text())

print(f"[OK] Saved time-to-first-policy Cox model -> {OUT_TTFP_COX}")
print(cox_fit.summary())


# ----------------------------
# 2) Robustness: OLS on mentioned-only papers
# Outcome=log(1 + days_to_first_policy_mention)
# Interpretation: negative coef on is_multiARC means shorter lag
# ----------------------------
ols_df = ttfp[ttfp["event"] == 1].copy()
ols_df["log_days"] = np.log1p(ols_df["t"].astype(float))

ols_fit = sm.OLS.from_formula(
#    "log_days ~ is_multiARC + OpenAccess_bin + pub_year_c + C(FoR)",
    "log_days ~ is_multiARC + OpenAccess_bin + pub_year_c",
    data=ols_df
).fit(cov_type="HC3")

with open(OUT_TTFP_OLS, "w", encoding="utf-8") as f:
    f.write("MODEL=OLS (mentioned-only)\n")
    f.write("Outcome=log(1 + days_to_first_policy_mention)\n\n")
    f.write(ols_fit.summary().as_text())

print(f"[OK] Saved time-to-first-policy OLS robustness model -> {OUT_TTFP_OLS}")
print(ols_fit.summary())


# ----------------------------
# Exposure-normalised policy
# ----------------------------
paper_level["policy_mentions_num"] = pd.to_numeric(paper_level["Policy mentions"], errors="coerce")
paper_level["policy_any"] = (paper_level["policy_mentions_num"].fillna(0) >= 1).astype(int)

# Policy mentions per year (exposure-normalised)
paper_level["policy_mentions_per_year"] = paper_level["policy_mentions_num"] / paper_level["years_since_pub"]

# Optional: cap extreme values from very recent papers (stability)
paper_level.loc[paper_level["years_since_pub"] < 0.5, "policy_mentions_per_year"] = np.nan

# Summaries
def group_summary(g):
    return pd.Series({
        "n_papers": len(g),
        "policy_rate": g["policy_any"].mean(),
        "mean_policy_mentions": g["policy_mentions_num"].mean(skipna=True),
        "median_citations": g["times_cited_num"].median(skipna=True),
        "mean_citations_per_year": g["citations_per_year"].mean(skipna=True),
    })

ms = paper_level[paper_level["k_arc"].isin([1])].copy()
mm = paper_level[paper_level["k_arc"] >= 2].copy()

multi_single_summary = pd.DataFrame({
    "group": ["single-ARC papers", "multi-ARC papers"],
    **pd.concat([group_summary(ms), group_summary(mm)], axis=1).T.set_index(pd.Index(["single-ARC papers", "multi-ARC papers"])).to_dict(orient="list")
})
multi_single_summary = pd.DataFrame({
    "group": ["single-ARC papers", "multi-ARC papers"],
    "n_papers": [len(ms), len(mm)],
    "policy_rate": [ms["policy_any"].mean(), mm["policy_any"].mean()],
    "mean_policy_mentions": [ms["policy_mentions_num"].mean(skipna=True), mm["policy_mentions_num"].mean(skipna=True)],
    "median_citations": [ms["times_cited_num"].median(skipna=True), mm["times_cited_num"].median(skipna=True)],
    "mean_citations_per_year": [ms["citations_per_year"].mean(skipna=True), mm["citations_per_year"].mean(skipna=True)],
})

multi_single_summary.to_csv(OUT_MULTI_SINGLE, index=False)
print(f"[OK] Saved multi vs single ARC paper summary -> {OUT_MULTI_SINGLE}")

# Plot: policy rate comparison
plt.figure(figsize=(7, 5))
plt.bar(multi_single_summary["group"], multi_single_summary["policy_rate"])
plt.ylabel("Proportion with ≥1 policy mention")
plt.title("Policy visibility: multi-ARC vs single-ARC papers")
plt.xticks(rotation=15, ha="right")
safe_savefig(FIG_MULTI_POLICY, FIG_MULTI_POLICY_PDF)
print(f"[OK] Saved multi vs single policy figure -> {FIG_MULTI_POLICY} (+ PDF)")

# Plot: citations distribution (boxplot) single vs multi
plt.figure(figsize=(8, 5))
plt.boxplot(
    [ms["times_cited_num"].dropna().values, mm["times_cited_num"].dropna().values],
    labels=["Single-ARC", "Multi-ARC"],
    showfliers=False
)
plt.ylabel("Times cited")
plt.title("Citations: multi-ARC vs single-ARC papers")
safe_savefig(FIG_MULTI_CITES, FIG_MULTI_CITES_PDF)
print(f"[OK] Saved multi vs single citations boxplot -> {FIG_MULTI_CITES} (+ PDF)")


# ----------------------------
# 6) Time trends (collaboration share, policy rate, OA share)
# ----------------------------
paper_level["pub_year_final"] = paper_level["pub_date_final"].dt.year
trend = paper_level.dropna(subset=["pub_year_final"]).copy()


# Limit to plausible window if you want:
trend = trend[(trend["pub_year_final"] >= 2010) & (trend["pub_year_final"] <= 2025)]

trend_year = trend.groupby("pub_year_final", as_index=False).agg(
    n_papers=("paper_id", "count"),
    share_multiARC=("is_multiARC", "mean"),
    policy_rate=("policy_any", "mean"),
    open_access_share=("Open Access", lambda s: (s.astype(str).str.strip() != "").mean()),
)

# Plot: share multi-ARC over time
plt.figure(figsize=(10, 5))
plt.plot(trend_year["pub_year_final"], trend_year["share_multiARC"])
plt.xlabel("Publication year")
plt.ylabel("Share of papers that are multi-ARC")
plt.title("Trend: cross-ARC collaboration over time")
safe_savefig(FIG_TREND_COLLAB, FIG_TREND_COLLAB_PDF)
print(f"[OK] Saved collaboration trend figure -> {FIG_TREND_COLLAB} (+ PDF)")

# Plot: policy rate over time
plt.figure(figsize=(10, 5))
plt.plot(trend_year["pub_year_final"], trend_year["policy_rate"])
plt.xlabel("Publication year")
plt.ylabel("Share with ≥1 policy mention")
plt.title("Trend: policy visibility over time")
safe_savefig(FIG_TREND_POLICY, FIG_TREND_POLICY_PDF)
print(f"[OK] Saved policy trend figure -> {FIG_TREND_POLICY} (+ PDF)")

# Plot: OA share over time
plt.figure(figsize=(10, 5))
plt.plot(trend_year["pub_year_final"], trend_year["open_access_share"])
plt.xlabel("Publication year")
plt.ylabel("Open access share")
plt.title("Trend: open access over time")
safe_savefig(FIG_TREND_OA, FIG_TREND_OA_PDF)
print(f"[OK] Saved open access trend figure -> {FIG_TREND_OA} (+ PDF)")

# ----------------------------
# Build model_df for policy regression
# ----------------------------
model_df = paper_level.copy()

model_df["pub_year_final"] = pd.to_numeric(model_df["pub_year_final"], errors="coerce")

model_df["FoR"] = model_df["Fields of Research (ANZSRC 2020)"].astype(str).fillna("unknown")
model_df["is_multiARC"] = (model_df["k_arc"] >= 2).astype(int)

model_df["policy_mentions_num"] = pd.to_numeric(model_df["Policy mentions"], errors="coerce")
model_df["policy_any"] = (model_df["policy_mentions_num"].fillna(0) >= 1).astype(int)

oa = model_df["Open Access"].astype(str).str.strip().str.lower()

# Clean categories
model_df["oa_type"] = np.select(
    [
        oa.str.contains(r"\bclosed\b", na=False),
        oa.str.contains(r"\bhybrid\b", na=False),
        oa.str.contains(r"\bgold\b", na=False),
        oa.str.contains(r"\boverall|all oa|alloa|open\b", na=False),
    ],
    [
        "closed",
        "hybrid",
        "gold",
        "open_other"
    ],
    default="unknown"
)

# Binary OA: 1 if not closed
model_df["OpenAccess_bin"] = np.where(model_df["oa_type"] == "closed", 0, 1)


# keep only usable rows
model_df = model_df.dropna(subset=["policy_any", "pub_year_final"])


# ----------------------------
# Multi vs Single ARC within 5-year cohorts
# ----------------------------

# Define five-year cohorts
def five_year_band(y):
    try:
        y = int(y)
    except:
        return np.nan
    if 2010 <= y <= 2014:
        return "2010-2014"
    if 2015 <= y <= 2019:
        return "2015-2019"
    if 2020 <= y <= 2024:
        return "2020-2024"
    return np.nan

paper_level["five_year_band"] = paper_level["pub_year_final"].apply(five_year_band)

# Keep only the analysis bands (optional)
paper_level_5yr = paper_level.dropna(subset=["five_year_band"]).copy()


OUT_MULTI_SINGLE_5YR = "multiARC_vs_singleARC_by_5year_band.csv"

# Define ARC-collaboration status
paper_level_5yr["is_multiARC"] = (paper_level_5yr["k_arc"] >= 2).astype(int)
paper_level_5yr["collab_group"] = np.where(paper_level_5yr["is_multiARC"] == 1, "multi-ARC", "single-ARC")

def summarise_group(g):
    return pd.Series({
        "n_papers": len(g),
        "policy_rate": g["policy_any"].mean(),
        "mean_policy_mentions_per_year": g["policy_mentions_per_year"].mean(skipna=True),
        "median_citations": g["times_cited_num"].median(skipna=True),
        "mean_citations_per_year": g["citations_per_year"].mean(skipna=True),
    })

multi_single_5yr = (
    paper_level_5yr
    .groupby(["five_year_band", "collab_group"])
    .apply(summarise_group)
    .reset_index()
)

multi_single_5yr.to_csv(OUT_MULTI_SINGLE_5YR, index=False)
print(f"[OK] Saved multi vs single ARC by cohort -> {OUT_MULTI_SINGLE_5YR}")

print("\nPreview: multi vs single within cohorts")
print(multi_single_5yr.to_string(index=False))

# ----------------------------
# ARC outcomes by 5-year cohort (fractional counting)
# ----------------------------
OUT_ARC_OUTCOMES_5YR = "arc_outcomes_by_5year_band.csv"

# Create band in df
df["five_year_band"] = df["pub_year_final"].apply(five_year_band)

df_5yr = df.dropna(subset=["five_year_band"]).copy()

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    if mask.sum() == 0:
        return np.nan
    return np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])

def weighted_rate(binary, weights):
    binary = np.asarray(binary, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(binary) & np.isfinite(weights)
    if mask.sum() == 0:
        return np.nan
    return np.sum(binary[mask] * weights[mask]) / np.sum(weights[mask])

rows = []
for (band, arc), sub in df_5yr.groupby(["five_year_band", "infra_name"]):
    w = sub["w_paper_to_arc"].astype(float)
    rows.append({
        "five_year_band": band,
        "infra_name": arc,
        "output_volume_fractional": float(w.sum()),
        "policy_rate_weighted": weighted_rate(sub["policy_any"], w),
        "policy_mentions_per_year_mean_weighted": weighted_mean(sub["policy_mentions_per_year"], w),
        "citations_per_year_mean_weighted": weighted_mean(sub["citations_per_year"], w),
    })

arc_out_5yr = pd.DataFrame(rows).sort_values(
    ["five_year_band", "output_volume_fractional"], ascending=[True, False]
)

arc_out_5yr.to_csv(OUT_ARC_OUTCOMES_5YR, index=False)
print(f"[OK] Saved ARC outcomes by 5-year cohort -> {OUT_ARC_OUTCOMES_5YR}")


# ============================================================
# EXTRA FIGURES: Ranked ARC outcome bars (paper-friendly)
# ============================================================

FIG_ARC_POLICY_RANKED = "fig_arc_policy_rate_ranked.png"
FIG_ARC_POLICY_RANKED_PDF = "fig_arc_policy_rate_ranked.pdf"

FIG_ARC_CITE_RATE_RANKED = "fig_arc_citations_per_year_ranked.png"
FIG_ARC_CITE_RATE_RANKED_PDF = "fig_arc_citations_per_year_ranked.pdf"

# Use arc_out (ARC outcomes dashboard) already built above
rank_df = arc_out.copy()

# ------------------------------------------------------------
# Figure A: Ranked policy visibility rate (weighted)
# ------------------------------------------------------------
rank_policy = rank_df.dropna(subset=["policy_rate_weighted"]).sort_values(
    "policy_rate_weighted", ascending=False
)

plt.figure(figsize=(11, 7))
plt.barh(rank_policy["infra_name"], rank_policy["policy_rate_weighted"])
plt.gca().invert_yaxis()
plt.xlabel("Weighted share of papers with ≥1 policy mention")
plt.ylabel("ARC")
plt.title("Policy visibility by ARC (fractionally counted)")

# Add small text labels for output volume (optional but nice)
for i, (_, r) in enumerate(rank_policy.iterrows()):
    plt.text(
        r["policy_rate_weighted"],
        i,
        f"  vol={r['output_volume_fractional']:.1f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()
plt.savefig(FIG_ARC_POLICY_RANKED, dpi=300)
plt.savefig(FIG_ARC_POLICY_RANKED_PDF)
plt.close()
print(f"[OK] Saved ranked policy figure -> {FIG_ARC_POLICY_RANKED} (+ PDF)")

# ------------------------------------------------------------
# Figure B: Ranked citations per year (age-normalised, weighted)
# ------------------------------------------------------------
rank_cites = rank_df.dropna(subset=["citations_per_year_mean_weighted"]).sort_values(
    "citations_per_year_mean_weighted", ascending=False
)

plt.figure(figsize=(11, 7))
plt.barh(rank_cites["infra_name"], rank_cites["citations_per_year_mean_weighted"])
plt.gca().invert_yaxis()
plt.xlabel("Weighted mean citations per year since publication")
plt.ylabel("ARC")
plt.title("Citation rate by ARC (age-normalised; fractionally counted)")

# Add volume labels
for i, (_, r) in enumerate(rank_cites.iterrows()):
    plt.text(
        r["citations_per_year_mean_weighted"],
        i,
        f"  vol={r['output_volume_fractional']:.1f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()
plt.savefig(FIG_ARC_CITE_RATE_RANKED, dpi=300)
plt.savefig(FIG_ARC_CITE_RATE_RANKED_PDF)
plt.close()
print(f"[OK] Saved ranked citations-per-year figure -> {FIG_ARC_CITE_RATE_RANKED} (+ PDF)")

# ============================================================
# FIGURE: Mean number of ARCs per paper over time
# ============================================================

FIG_MEAN_ARCS_PER_PAPER = "fig_trend_mean_ARCs_per_paper.png"
FIG_MEAN_ARCS_PER_PAPER_PDF = "fig_trend_mean_ARCs_per_paper.pdf"

# Keep valid years and ARC counts
tmp = paper_level.dropna(subset=["pub_year_final"]).copy()
tmp["pub_year_final"] = pd.to_numeric(tmp["pub_year_final"], errors="coerce")
tmp = tmp[(tmp["pub_year_final"] >= 2010) & (tmp["pub_year_final"] <= 2025)]

# Ensure k_arc exists
if "k_arc" not in tmp.columns:
    raise ValueError("k_arc not found in paper_level. Merge paper_k first to get number of ARCs per paper.")

# Mean ARCs per paper by year
mean_k_by_year = (
    tmp.groupby("pub_year_final", as_index=False)
       .agg(mean_ARCs_per_paper=("k_arc", "mean"),
            median_ARCs_per_paper=("k_arc", "median"),
            n_papers=("paper_id", "count"))
)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(mean_k_by_year["pub_year_final"], mean_k_by_year["mean_ARCs_per_paper"], marker="o")
plt.xlabel("Publication year")
plt.ylabel("Mean number of ARCs per paper")
plt.title("Trend: collaboration intensity (mean ARCs per paper) over time")

# Optional: annotate small-n years (helps interpretation)
for _, r in mean_k_by_year.iterrows():
    if r["n_papers"] < 50:  # tweak if needed
        plt.annotate(f"n={int(r['n_papers'])}",
                     (r["pub_year_final"], r["mean_ARCs_per_paper"]),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(FIG_MEAN_ARCS_PER_PAPER, dpi=300)
plt.savefig(FIG_MEAN_ARCS_PER_PAPER_PDF)
plt.close()

print(f"[OK] Saved mean ARCs-per-paper trend -> {FIG_MEAN_ARCS_PER_PAPER} (+ PDF)")

# ----------------------------
# POLICY MODEL (logistic): policy_any ~ multiARC + year + OA + FoR
# ----------------------------

#run the following six lines once only as a check
print("Policy_any value counts:")
print(model_df["policy_any"].value_counts(dropna=False))

print("\nOpenAccess_bin value counts:")
print(model_df["OpenAccess_bin"].value_counts(dropna=False))

print("\nMultiARC value counts:")
print(model_df["is_multiARC"].value_counts(dropna=False))
# once-only lines end here




OUT_POLICY_MODEL = "model_policy_any_logit.txt"

method_used = None
policy_fit = None

# ---- 1) Full model attempt (may fail)
try:
    policy_fit = smf.logit(
        "policy_any ~ is_multiARC + OpenAccess_bin + C(pub_year_final) + C(FoR)",
        data=model_df
    ).fit(disp=False)
    method_used = "logit_full"

except Exception as e1:
    print("[WARN] Full logit failed:", repr(e1))
    print("[INFO] Trying simpler year-trend model...")

    # ---- 2) Simple model attempt
    try:
        policy_fit = smf.logit(
            "policy_any ~ is_multiARC + OpenAccess_bin + pub_year_final",
            data=model_df
        ).fit(disp=False)
        method_used = "logit_simple_year_trend"

    except Exception as e2:
        print("[WARN] Simple logit also failed:", repr(e2))
        print("[INFO] Using regularised logistic regression (L2)...")

        # ---- 3) Regularised model (almost always works)
        policy_fit = smf.logit(
            "policy_any ~ is_multiARC + OpenAccess_bin + pub_year_final",
            data=model_df
        ).fit_regularized(alpha=1.0, L1_wt=0.0)  # L2 penalty
        method_used = "logit_regularized_L2_alpha1"

with open(OUT_POLICY_MODEL, "w", encoding="utf-8") as f:
    f.write(f"METHOD_USED={method_used}\n\n")
    f.write(policy_fit.summary().as_text())

print(f"[OK] Saved policy model output -> {OUT_POLICY_MODEL} (method={method_used})")
print(policy_fit.summary())


# ----------------------------
# CITATIONS MODEL (Discrete NB2 with exposure offset)
# ----------------------------
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

OUT_CITATIONS_MODEL = "model_citations_nb_offset.txt"

cit_df = paper_level.copy()

cit_df["times_cited_num"] = pd.to_numeric(cit_df["Times cited"], errors="coerce")
cit_df["pub_year_final"]  = pd.to_numeric(cit_df["pub_year_final"], errors="coerce")
cit_df["is_multiARC"]     = (cit_df["k_arc"] >= 2).astype(int)

# OA binary (closed=0 else=1)
oa = cit_df["Open Access"].astype(str).str.strip().str.lower()
cit_df["OpenAccess_bin"] = np.where(oa.str.contains(r"\bclosed\b", na=False), 0, 1)

# Keep valid rows
cit_df = cit_df.dropna(subset=["times_cited_num", "years_since_pub", "pub_year_final"])
cit_df = cit_df[np.isfinite(cit_df["times_cited_num"]) & np.isfinite(cit_df["years_since_pub"])]
cit_df = cit_df[(cit_df["times_cited_num"] >= 0) & (cit_df["years_since_pub"] > 0)]

# Offset must exist
cit_df["log_exposure"] = np.log(cit_df["years_since_pub"])
cit_df = cit_df[np.isfinite(cit_df["log_exposure"])]

print("[INFO] cit_df rows for discrete NB2:", len(cit_df))

# Optional: cap extreme citation outliers (modelling only)
cap = cit_df["times_cited_num"].quantile(0.995)
cit_df["times_cited_num"] = cit_df["times_cited_num"].clip(upper=cap)

# Center year for stability
cit_df["pub_year_c"] = cit_df["pub_year_final"] - cit_df["pub_year_final"].mean()

X = cit_df[["is_multiARC", "OpenAccess_bin", "pub_year_c"]].copy()
X = sm.add_constant(X, has_constant="add")
y = cit_df["times_cited_num"].astype(float)

nb2_fit = NegativeBinomial(y, X, offset=cit_df["log_exposure"]).fit(disp=False, maxiter=200)

with open(OUT_CITATIONS_MODEL, "w", encoding="utf-8") as f:
    f.write("METHOD_USED=discrete_NB2_offset\n")
    f.write(f"CITATION_CAP_0.995={cap}\n\n")
    f.write(nb2_fit.summary().as_text())

print(f"[OK] Saved citations model output -> {OUT_CITATIONS_MODEL}")
print(nb2_fit.summary())






# ----------------------------
# POLICY COUNT MODEL (Discrete NB2 with exposure offset): policy_mentions ~ predictors + offset(log(years_since_pub))
# ----------------------------
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

OUT_POLICY_COUNT_MODEL = "model_policy_mentions_nb_offset.txt"

pc_df = paper_level.copy()

# Outcome + predictors
pc_df["policy_mentions_num"] = pd.to_numeric(pc_df["Policy mentions"], errors="coerce")
pc_df["pub_year_final"]      = pd.to_numeric(pc_df["pub_year_final"], errors="coerce")
pc_df["is_multiARC"]         = (pc_df["k_arc"] >= 2).astype(int)

# Improved OA (closed=0 else=1)
oa = pc_df["Open Access"].astype(str).str.strip().str.lower()
pc_df["OpenAccess_bin"] = np.where(oa.str.contains(r"\bclosed\b", na=False), 0, 1)

# Keep valid rows (need exposure + nonnegative counts)
pc_df = pc_df.dropna(subset=["policy_mentions_num", "years_since_pub", "pub_year_final"])
pc_df = pc_df[np.isfinite(pc_df["policy_mentions_num"]) & np.isfinite(pc_df["years_since_pub"])]
pc_df = pc_df[(pc_df["policy_mentions_num"] >= 0) & (pc_df["years_since_pub"] > 0)]

# Offset
pc_df["log_exposure"] = np.log(pc_df["years_since_pub"])
pc_df = pc_df[np.isfinite(pc_df["log_exposure"])]

print("[INFO] pc_df rows for discrete NB2:", len(pc_df))

# Policy mentions are extremely skewed → cap outliers for stability (modelling only)
cap = pc_df["policy_mentions_num"].quantile(0.995)
pc_df["policy_mentions_num"] = pc_df["policy_mentions_num"].clip(upper=cap)

# Center year for stability
pc_df["pub_year_c"] = pc_df["pub_year_final"] - pc_df["pub_year_final"].mean()

# Build design matrix
X = pc_df[["is_multiARC", "OpenAccess_bin", "pub_year_c"]].copy()
X = sm.add_constant(X, has_constant="add")
y = pc_df["policy_mentions_num"].astype(float)

# Fit discrete NB2 with offset
pc_nb2 = NegativeBinomial(y, X, offset=pc_df["log_exposure"]).fit(disp=False, maxiter=200)

with open(OUT_POLICY_COUNT_MODEL, "w", encoding="utf-8") as f:
    f.write("METHOD_USED=discrete_NB2_offset\n")
    f.write(f"POLICY_CAP_0.995={cap}\n\n")
    f.write(pc_nb2.summary().as_text())

print(f"[OK] Saved policy mentions NB2-offset model -> {OUT_POLICY_COUNT_MODEL}")
print(pc_nb2.summary())


# date checks
print(
    paper_level["pub_year_final"]
    .value_counts(dropna=False)
    .sort_index()
)

print("Unique papers:", df["paper_id"].nunique())
print("Rows in long df (paper×ARC links):", len(df))
print("Mean ARC links per paper:", len(df) / df["paper_id"].nunique())

import numpy as np
import pandas as pd

# ----------------------------
# 1) Define funding eras + totals
# ----------------------------
ERA_FUNDS = {
    "2010-2014_CLAHRC": 88_000_000,
    "2015-2019_ARC":    144_852_534,
    "2020-2025_ARC":    202_535_848,
}

def assign_era(y):
    if pd.isna(y):
        return np.nan
    y = int(y)
    if 2010 <= y <= 2014:
        return "2010-2014_CLAHRC"
    if 2015 <= y <= 2019:
        return "2015-2019_ARC"
    if 2020 <= y <= 2025:
        return "2020-2025_ARC"
    return np.nan

paper_level["era"] = paper_level["pub_year_final"].apply(assign_era)

# ----------------------------
# 2) Paper-level outcomes (clean)
# ----------------------------
paper_level["times_cited_num"] = pd.to_numeric(paper_level["Times cited"], errors="coerce")
paper_level["policy_mentions_num"] = pd.to_numeric(paper_level["Policy mentions"], errors="coerce")
paper_level["policy_any"] = (paper_level["policy_mentions_num"].fillna(0) >= 1).astype(int)

# ----------------------------
# 3) Fractional ARC attribution table
#    expects a column like: "ARCs" = "ARC West | ARC Wessex" or "0"
# ----------------------------
def split_arcs(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "" or s == "0":
        return []
    return [a.strip() for a in s.split("|") if a.strip()]

tmp = paper_level[["DOI", "era", "times_cited_num", "policy_mentions_num", "policy_any", "ARCs"]].copy()
tmp["arc_list"] = tmp["ARCs"].apply(split_arcs)
tmp["k"] = tmp["arc_list"].apply(len)
tmp = tmp[tmp["era"].notna()].copy()

# explode
arc_long = tmp.explode("arc_list").rename(columns={"arc_list": "ARC"})
arc_long = arc_long[arc_long["ARC"].notna()].copy()
arc_long["k"] = arc_long.groupby("DOI")["ARC"].transform("nunique")
arc_long["w_frac"] = 1 / arc_long["k"]

# ----------------------------
# 4) ERA-level value-for-money (system totals)
# ----------------------------
era_totals = tmp.groupby("era", as_index=False).agg(
    papers=("DOI", "nunique"),
    citations=("times_cited_num", "sum"),
    policy_mentions=("policy_mentions_num", "sum"),
    policy_any_papers=("policy_any", "sum")
)

era_totals["funding"] = era_totals["era"].map(ERA_FUNDS)
era_totals["papers_per_£m"] = era_totals["papers"] / (era_totals["funding"] / 1e6)
era_totals["citations_per_£m"] = era_totals["citations"] / (era_totals["funding"] / 1e6)
era_totals["policy_mentions_per_£m"] = era_totals["policy_mentions"] / (era_totals["funding"] / 1e6)
era_totals["policy_any_papers_per_£m"] = era_totals["policy_any_papers"] / (era_totals["funding"] / 1e6)

era_totals.to_csv("outputs_per_pound_era.csv", index=False)

# ----------------------------
# 4b) YEAR-level value-for-money (system totals)
#     Allocate era funding evenly across years in that era
# ----------------------------

# Build a per-year funding table from your era totals
YEAR_FUNDS = {}

# 2010–2014 (5 years)
for y in range(2010, 2015):
    YEAR_FUNDS[y] = 88_000_000 / 5

# 2015–2019 (5 years)
for y in range(2015, 2020):
    YEAR_FUNDS[y] = 144_852_534 / 5

# 2020–2025 (6 years in your data; funding runs to 2026 but you don't have 2026 pubs)
for y in range(2020, 2026):
    YEAR_FUNDS[y] = 202_535_848 / 6


# System totals by year
year_totals = tmp.copy()
year_totals["pub_year_final"] = pd.to_numeric(paper_level["pub_year_final"], errors="coerce")

year_totals = year_totals.dropna(subset=["pub_year_final"])
year_totals["pub_year_final"] = year_totals["pub_year_final"].astype(int)

year_totals = year_totals.groupby("pub_year_final", as_index=False).agg(
    papers=("DOI", "nunique"),
    citations=("times_cited_num", "sum"),
    policy_mentions=("policy_mentions_num", "sum"),
    policy_any_papers=("policy_any", "sum")
).rename(columns={"pub_year_final": "year"})

# Add allocated funding per year
year_totals["funding_allocated"] = year_totals["year"].map(YEAR_FUNDS)

# Only keep years where we have a funding allocation
year_totals = year_totals[year_totals["funding_allocated"].notna()].copy()

# Outputs per £m
year_totals["papers_per_£m"] = year_totals["papers"] / (year_totals["funding_allocated"] / 1e6)
year_totals["citations_per_£m"] = year_totals["citations"] / (year_totals["funding_allocated"] / 1e6)
year_totals["policy_mentions_per_£m"] = year_totals["policy_mentions"] / (year_totals["funding_allocated"] / 1e6)
year_totals["policy_any_papers_per_£m"] = year_totals["policy_any_papers"] / (year_totals["funding_allocated"] / 1e6)

year_totals.to_csv("outputs_per_pound_year.csv", index=False)

print("[OK] Saved: outputs_per_pound_year.csv")

# ----------------------------
# Figure: Policy mentions per £m over time (system total)
# ----------------------------
import matplotlib.pyplot as plt

policy_cash_time_png = "fig_policy_mentions_per_pound_by_year.png"
policy_cash_time_pdf = "fig_policy_mentions_per_pound_by_year.pdf"

plot_df = year_totals.sort_values("year").copy()

plt.figure(figsize=(9, 5))
plt.plot(plot_df["year"], plot_df["policy_mentions_per_£m"], marker="o")
plt.xlabel("Publication year")
plt.ylabel("Policy mentions per £m (allocated funding)")
plt.title("Policy reach per £m over time (system total)")
plt.tight_layout()
plt.savefig(policy_cash_time_png, dpi=300)
plt.savefig(policy_cash_time_pdf)
plt.close()

print(f"[OK] Saved policy by cash over time figure -> {policy_cash_time_png} and {policy_cash_time_pdf}")


# ----------------------------
# 5) ARC-level value-for-money within each era (fractional)
# ----------------------------
arc_totals = arc_long.groupby(["era", "ARC"], as_index=False).agg(
    frac_papers=("w_frac", "sum"),
    frac_citations=("times_cited_num", lambda s: np.nansum(s)),     # still paper-level counts; OK for totals
    frac_policy_mentions=("policy_mentions_num", lambda s: np.nansum(s)),
    frac_policy_any=("policy_any", lambda s: np.nansum(s)),
)

arc_totals["funding"] = arc_totals["era"].map(ERA_FUNDS)
arc_totals["frac_papers_per_£m"] = arc_totals["frac_papers"] / (arc_totals["funding"] / 1e6)
arc_totals["frac_policy_mentions_per_£m"] = arc_totals["frac_policy_mentions"] / (arc_totals["funding"] / 1e6)

arc_totals.to_csv("outputs_per_pound_arc_fractional.csv", index=False)

print("[OK] Saved: outputs_per_pound_era.csv and outputs_per_pound_arc_fractional.csv")

print("\nDone ?")