"""
Tactical CBM Engine - Architecture Rationale Verification Tests (Phase 8 / Slice 3)
Authors: Senior ML/DL Architect & Gravitas Reliability Runtime

Validates:
  1. docs/ARCHITECTURE_RATIONALE.md exists and meets completeness criteria.
  2. All 4 required pillars (B1, B2, B3, B4) are formally articulated.
  3. Strict zero-celebrity player names constraint is maintained.
  4. Mathematical formulations and empirical citations are present.
  5. The 4 formal architectural invariants are documented.
"""

import os
import sys
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RATIONALE_PATH = BASE_DIR / "docs" / "ARCHITECTURE_RATIONALE.md"

PROHIBITED_CELEBRITY_NAMES = [
    "messi", "ronaldo", "mbappe", "haaland", "neymar",
    "debruyne", "de bruyne", "kane", "bellingham", "salah",
    "lewandowski", "vinicius", "modric", "kroos"
]


class TestArchitectureRationale:

    def test_manifesto_file_exists_and_substantial(self):
        """Verifies docs/ARCHITECTURE_RATIONALE.md exists and is a substantive technical document."""
        assert RATIONALE_PATH.exists(), f"Missing manifesto file at {RATIONALE_PATH}"
        text = RATIONALE_PATH.read_text(encoding="utf-8")
        assert len(text) >= 5000, f"Manifesto too brief ({len(text)} characters), expected >= 5000"

    def test_manifesto_contains_all_four_pillars(self):
        """Verifies all four architectural justification sections (B1-B4) are present."""
        text = RATIONALE_PATH.read_text(encoding="utf-8")
        assert "Delaunay Triangulation vs. Metric Graph Heuristics" in text, "Missing B1 section"
        assert "Concept Head Topology & Bottleneck Constraints" in text, "Missing B2 section"
        assert "Node Feature Selection & Anonymization" in text, "Missing B3 section"
        assert "Spatio-Temporal Kinematics vs. Heavy Temporal Modeling" in text, "Missing B4 section"

    def test_manifesto_contains_empirical_citations(self):
        """Verifies empirical benchmark tables are cross-referenced."""
        text = RATIONALE_PATH.read_text(encoding="utf-8")
        assert "results/graph_topology_table.md" in text, "Missing graph topology benchmark citation"
        assert "results/hyperparameter_table.md" in text, "Missing hyperparameter sweep citation"
        assert "results/ablation_table.md" in text, "Missing ablation table citation"

    def test_manifesto_zero_celebrity_names_invariant(self):
        """Enforces that zero celebrity player names appear in the architecture manifesto."""
        text = RATIONALE_PATH.read_text(encoding="utf-8").lower()
        for name in PROHIBITED_CELEBRITY_NAMES:
            # Check only whole word occurrences
            assert f" {name} " not in text and f"({name})" not in text and f"`{name}`" not in text, \
                f"Prohibited celebrity player name '{name}' detected in manifesto"

    def test_manifesto_invariants_declaration(self):
        """Verifies formal declaration of all four core architectural invariants."""
        text = RATIONALE_PATH.read_text(encoding="utf-8")
        assert "INVARIANT 1: DELAUNAY ADJACENCY" in text
        assert "INVARIANT 2: ZERO LATENT SKIP TO THREAT" in text
        assert "INVARIANT 3: PLAYER ANONYMIZATION" in text
        assert "INVARIANT 4: REAL-TIME INFERENCE BUDGET" in text
