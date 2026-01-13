"""
kb_interface.py - Read-Only Interface to the Knowledge Base

This module provides a deterministic, read-only interface to the KB.
The KB is the single source of truth. No mutation is allowed.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List


class KnowledgeBase:
    """
    Immutable, read-only Knowledge Base interface.
    All artifacts are loaded once and never modified at runtime.
    """

    def __init__(self, kb_root: str):
        self.kb_root = Path(kb_root)
        self._strategies: Dict[str, Dict] = {}
        self._indicators: Dict[str, Dict] = {}
        self._datasets: Dict[str, Dict] = {}
        self._load_all()

    def _load_json(self, filepath: Path) -> Optional[Dict]:
        """Load and parse a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"[KB ERROR] Failed to load {filepath}: {e}")
            return None

    def _load_all(self):
        """Load all KB artifacts into memory."""
        # Load strategies
        strategies_dir = self.kb_root / "strategies"
        if strategies_dir.exists():
            for f in strategies_dir.glob("*.json"):
                data = self._load_json(f)
                if data and "id" in data:
                    self._strategies[data["id"]] = data

        # Load indicators
        indicators_dir = self.kb_root / "indicators"
        if indicators_dir.exists():
            for f in indicators_dir.glob("*.json"):
                data = self._load_json(f)
                if data and "id" in data:
                    self._indicators[data["id"]] = data

        # Load datasets
        datasets_dir = self.kb_root / "datasets"
        if datasets_dir.exists():
            for f in datasets_dir.glob("*.json"):
                data = self._load_json(f)
                if data and "id" in data:
                    self._datasets[data["id"]] = data

    # -------------------------------------------------------------------------
    # READ-ONLY ACCESS METHODS
    # -------------------------------------------------------------------------

    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Retrieve a strategy by its ID. Returns None if not found."""
        return self._strategies.get(strategy_id)

    def get_indicator(self, indicator_id: str) -> Optional[Dict]:
        """Retrieve an indicator by its ID. Returns None if not found."""
        return self._indicators.get(indicator_id)

    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Retrieve a dataset by its ID. Returns None if not found."""
        return self._datasets.get(dataset_id)

    def list_strategies(self) -> List[str]:
        """List all available strategy IDs."""
        return list(self._strategies.keys())

    def list_indicators(self) -> List[str]:
        """List all available indicator IDs."""
        return list(self._indicators.keys())

    def list_datasets(self) -> List[str]:
        """List all available dataset IDs."""
        return list(self._datasets.keys())

    def search_by_name(self, name_query: str, artifact_type: str = "all") -> List[Dict]:
        """
        Search artifacts by name (case-insensitive partial match).
        artifact_type: 'strategy', 'indicator', 'dataset', or 'all'
        """
        results = []
        query_lower = name_query.lower()

        if artifact_type in ("strategy", "all"):
            for s in self._strategies.values():
                if query_lower in s.get("name", "").lower():
                    results.append({"type": "strategy", **s})

        if artifact_type in ("indicator", "all"):
            for i in self._indicators.values():
                if query_lower in i.get("name", "").lower():
                    results.append({"type": "indicator", **i})

        if artifact_type in ("dataset", "all"):
            for d in self._datasets.values():
                if query_lower in d.get("name", "").lower():
                    results.append({"type": "dataset", **d})

        return results

    def verify_hash(self, artifact_id: str, artifact_type: str) -> bool:
        """
        Verify the hash of an artifact (placeholder for demo).
        In production, this would recompute the hash from source.
        """
        # For demo, we just check if the hash field exists
        artifact = None
        if artifact_type == "strategy":
            artifact = self._strategies.get(artifact_id)
        elif artifact_type == "indicator":
            artifact = self._indicators.get(artifact_id)
        elif artifact_type == "dataset":
            artifact = self._datasets.get(artifact_id)

        return artifact is not None and "hash" in artifact
