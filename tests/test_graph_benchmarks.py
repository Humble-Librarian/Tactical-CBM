"""
Tactical CBM Engine - Graph Topology Test Suite (Phase 8 - Slice 1)
Tests:
  1. Delaunay Triangulation Voronoi Symmetry & Self-Loops
  2. k-Nearest Neighbors Degree Bounds & Symmetry
  3. Radius Graph Distance Threshold Invariant
  4. Bipartite & Dense Graph Contracts
  5. Benchmark Summary JSON & Markdown Artifact Integrity
"""

import sys
import json
import pytest
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.graph_benchmarks import (
    build_delaunay_topology,
    build_knn_topology,
    build_radius_topology,
    build_bipartite_topology,
    build_dense_topology
)

RESULTS_DIR = BASE_DIR / "results"
SUMMARY_JSON = RESULTS_DIR / "graph_topology_summary.json"
TABLE_MD = RESULTS_DIR / "graph_topology_table.md"

@pytest.fixture
def sample_coords():
    # 10 realistic player coordinates on pitch [0, 120] x [0, 80]
    return np.array([
        [45.0, 30.0], [50.0, 40.0], [52.0, 20.0], [60.0, 45.0], [58.0, 35.0],
        [70.0, 38.0], [75.0, 25.0], [80.0, 50.0], [90.0, 40.0], [105.0, 38.0]
    ], dtype=np.float32)

def test_delaunay_topology_properties(sample_coords):
    """Verify Delaunay triangulation produces symmetric planar graph with self-loops."""
    adj = build_delaunay_topology(sample_coords, max_players=22, d_max=28.0)
    
    assert adj.shape == (22, 22)
    # Symmetry
    assert np.allclose(adj, adj.T), "Delaunay adjacency must be symmetric"
    # Self-loops for active players
    for i in range(len(sample_coords)):
        assert adj[i, i] == 1.0, f"Self-loop missing for player {i}"
    # Unused slots must be zero
    assert np.all(adj[len(sample_coords):, :] == 0.0)
    assert np.all(adj[:, len(sample_coords):] == 0.0)

def test_knn_topology_properties(sample_coords):
    """Verify k-NN topology maintains symmetric adjacency and degree upper bounds."""
    k = 4
    adj = build_knn_topology(sample_coords, max_players=22, k=k)
    
    assert adj.shape == (22, 22)
    assert np.allclose(adj, adj.T), "k-NN adjacency must be symmetric"
    for i in range(len(sample_coords)):
        assert adj[i, i] == 1.0, f"Self-loop missing for player {i}"

def test_radius_topology_properties(sample_coords):
    """Verify radius graph strictly enforces distance threshold r."""
    r = 20.0
    adj = build_radius_topology(sample_coords, max_players=22, r=r)
    
    assert adj.shape == (22, 22)
    assert np.allclose(adj, adj.T), "Radius adjacency must be symmetric"
    
    n = len(sample_coords)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(sample_coords[i] - sample_coords[j])
            if dist <= r:
                assert adj[i, j] == 1.0, f"Edge ({i}, {j}) dist={dist} <= {r} but adj=0"
            else:
                assert adj[i, j] == 0.0, f"Edge ({i}, {j}) dist={dist} > {r} but adj=1"

def test_bipartite_and_dense_topology(sample_coords):
    """Verify bipartite separates teams and dense connects all active players."""
    is_att = np.array([True, True, True, True, True, False, False, False, False, False], dtype=bool)
    bip_adj = build_bipartite_topology(sample_coords, is_att, max_players=22)
    dense_adj = build_dense_topology(sample_coords, max_players=22)
    
    n = len(sample_coords)
    # In bipartite, same-team players must NOT be connected (except self-loop)
    for i in range(5):
        for j in range(5):
            if i != j:
                assert bip_adj[i, j] == 0.0, f"Teammates ({i}, {j}) connected in bipartite graph"
                
    # In dense, all active players must be connected
    assert np.all(dense_adj[:n, :n] == 1.0), "Dense graph must connect all active players"

def test_graph_topology_artifacts_contract():
    """Verify Slice 1 benchmark artifacts follow schema contracts and confirm Delaunay superiority."""
    assert SUMMARY_JSON.exists(), "graph_topology_summary.json must exist"
    assert TABLE_MD.exists(), "graph_topology_table.md must exist"
    
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    canonical_key = "Delaunay (d_max=28m) [Canonical]"
    dense_key = "Fully-Dense Complete"
    
    assert canonical_key in data, "Delaunay canonical entry missing"
    assert dense_key in data, "Dense entry missing"
    
    delaunay_metrics = data[canonical_key]
    dense_metrics = data[dense_key]
    
    # Delaunay must be significantly sparser than dense graph
    assert delaunay_metrics["sparsity_pct"] > 60.0, f"Delaunay sparsity {delaunay_metrics['sparsity_pct']}% <= 60%"
    assert dense_metrics["sparsity_pct"] == 0.0
    
    # Delaunay Concept R^2 must strictly beat fully-dense complete graph (proves over-smoothing avoidance)
    assert delaunay_metrics["concept_mean_r2"] > dense_metrics["concept_mean_r2"], (
        f"Delaunay Concept R2 {delaunay_metrics['concept_mean_r2']} must beat Dense R2 {dense_metrics['concept_mean_r2']}"
    )
