from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config
from src.data.schema import DataSchema, default_schema


@dataclass
class GroupStats:
    pairs: pd.DataFrame  # columns: from_state, to_state, ead_sum, n_obs
    n_obs: int
    ead_sum: float


class TransitionModel:
    def __init__(
        self,
        df: pd.DataFrame,
        schema: Optional[DataSchema] = None,
        state_order: Optional[List[str]] = None,
        min_obs: Optional[int] = None,
        min_ead: Optional[float] = None,
        weight_mode: Optional[str] = None,
        smoothing_cfg: Optional[dict] = None,
        fallback_order: Optional[List[str]] = None,
        absorbing_states: Optional[Sequence[str]] = None,
        closed_absorbing: Optional[bool] = None,
    ) -> None:
        self.schema = schema or default_schema()
        self.state_order = state_order or config.STATE_ORDER
        self.state_index = {s: i for i, s in enumerate(self.state_order)}
        self.segment_cols = list(self.schema.segment_cols)
        self.min_obs = min_obs if min_obs is not None else config.MIN_OBS
        self.min_ead = min_ead if min_ead is not None else config.MIN_EAD
        self.weight_mode = (weight_mode or config.TRANSITION_WEIGHT_MODE).lower()
        self.smoothing_cfg = smoothing_cfg or config.SMOOTHING
        self.fallback_order = fallback_order or config.FALLBACK_ORDER
        self.absorbing_states = list(absorbing_states) if absorbing_states is not None else list(config.ABSORBING_STATES)
        self.closed_absorbing = closed_absorbing if closed_absorbing is not None else config.CLOSED_ABSORBING

        if self.weight_mode not in {"ead", "count"}:
            raise ValueError(f"Unsupported weight_mode: {self.weight_mode}")

        self.transition_pairs = self._build_transition_pairs(df)
        self.segment_mob_stats = self._aggregate_stats(self.segment_cols + ["from_mob"])
        self.segment_all_stats = self._aggregate_stats(self.segment_cols)
        self.global_mob_stats = self._aggregate_stats(["from_mob"])
        self.global_all_stats = self._aggregate_stats([])

    def _build_transition_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        ordered = df.sort_values([self.schema.id_col, self.schema.mob_col, self.schema.date_col])
        ordered["to_state"] = ordered.groupby(self.schema.id_col)[self.schema.state_col].shift(-1)
        ordered["to_mob"] = ordered.groupby(self.schema.id_col)[self.schema.mob_col].shift(-1)
        mask = ordered["to_state"].notna() & (ordered["to_mob"] == ordered[self.schema.mob_col] + 1)
        pairs = ordered.loc[mask, self.segment_cols + [self.schema.mob_col, self.schema.state_col, "to_state", self.schema.ead_col]]
        pairs = pairs.rename(
            columns={
                self.schema.mob_col: "from_mob",
                self.schema.state_col: "from_state",
                "to_state": "to_state",
                self.schema.ead_col: "ead_from",
            }
        )
        pairs["n_obs"] = 1
        return pairs.reset_index(drop=True)

    def _aggregate_stats(self, group_cols: List[str]) -> Dict[Tuple, GroupStats]:
        if self.transition_pairs.empty:
            return {}

        agg_cols = group_cols + ["from_state", "to_state"]
        grouped = (
            self.transition_pairs.groupby(agg_cols, dropna=False)
            .agg(ead_sum=("ead_from", "sum"), n_obs=("n_obs", "sum"))
            .reset_index()
        )

        stats: Dict[Tuple, GroupStats] = {}
        if not group_cols:
            total_ead = float(grouped["ead_sum"].sum())
            total_obs = int(grouped["n_obs"].sum())
            stats[tuple()] = GroupStats(
                pairs=grouped[["from_state", "to_state", "ead_sum", "n_obs"]].reset_index(drop=True),
                n_obs=total_obs,
                ead_sum=total_ead,
            )
            return stats

        summary = grouped.groupby(group_cols, dropna=False).agg(
            ead_sum=("ead_sum", "sum"), n_obs=("n_obs", "sum")
        )

        for key, group_df in grouped.groupby(group_cols, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            totals = summary.loc[key]
            stats[key_tuple] = GroupStats(
                pairs=group_df[["from_state", "to_state", "ead_sum", "n_obs"]].reset_index(drop=True),
                n_obs=int(totals["n_obs"]),
                ead_sum=float(totals["ead_sum"]),
            )
        return stats

    def _passes_threshold(self, stats: GroupStats) -> bool:
        return stats.n_obs >= self.min_obs and stats.ead_sum >= self.min_ead

    def _get_weight_col(self) -> str:
        return "ead_sum" if self.weight_mode == "ead" else "n_obs"

    def _select_candidate(self, segment_key: Tuple, mob: int) -> Tuple[Optional[GroupStats], str, Optional[int]]:
        for level in self.fallback_order:
            if level == "segment_mob":
                key = (*segment_key, mob)
                stats = self.segment_mob_stats.get(key)
                if stats and self._passes_threshold(stats):
                    return stats, level, mob
            elif level == "segment_nearest_previous_mob":
                candidates = [
                    (k[-1], v) for k, v in self.segment_mob_stats.items() if k[:-1] == segment_key and k[-1] < mob and self._passes_threshold(v)
                ]
                if candidates:
                    chosen_mob, stats = sorted(candidates, key=lambda kv: kv[0], reverse=True)[0]
                    return stats, level, chosen_mob
            elif level == "segment_all":
                stats = self.segment_all_stats.get(segment_key)
                if stats and self._passes_threshold(stats):
                    return stats, level, None
            elif level == "global_mob":
                stats = self.global_mob_stats.get((mob,))
                if stats and self._passes_threshold(stats):
                    return stats, level, mob
            elif level == "global_all":
                stats = self.global_all_stats.get(tuple())
                if stats and self._passes_threshold(stats):
                    return stats, level, None
            elif level == "identity":
                break
        return None, "identity", mob

    def _build_probability_matrix(
        self,
        stats: GroupStats,
        parent_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_states = len(self.state_order)
        matrix = np.zeros((n_states, n_states))
        weight_col = self._get_weight_col()
        weight_lookup = {
            (row.from_state, row.to_state): getattr(row, weight_col) for row in stats.pairs.itertuples(index=False)
        }
        row_totals: Dict[str, float] = {}
        for (from_state, _), weight in weight_lookup.items():
            row_totals[from_state] = row_totals.get(from_state, 0.0) + weight

        strength = float(self.smoothing_cfg.get("strength", 0.0))
        min_prob = float(self.smoothing_cfg.get("min_prob", 0.0))

        for i, from_state in enumerate(self.state_order):
            row_vals = np.array([weight_lookup.get((from_state, to_state), 0.0) for to_state in self.state_order], dtype=float)
            total = float(row_totals.get(from_state, 0.0))
            if parent_matrix is not None and strength > 0:
                parent_row = parent_matrix[i]
                prob_row = (row_vals + strength * parent_row) / (total + strength)
            elif total > 0:
                prob_row = row_vals / total
            else:
                prob_row = np.zeros_like(row_vals)

            if prob_row.sum() == 0 and min_prob > 0:
                prob_row = np.array([min_prob] * len(self.state_order))

            if prob_row.sum() > 0:
                prob_row = prob_row / prob_row.sum()
            matrix[i, :] = prob_row

        matrix = self._apply_absorbing(matrix)
        return matrix

    def _apply_absorbing(self, matrix: np.ndarray) -> np.ndarray:
        absorbing = set(self.absorbing_states)
        if not self.closed_absorbing and "CLOSED" in absorbing:
            absorbing.remove("CLOSED")
        for state in absorbing:
            if state in self.state_index:
                idx = self.state_index[state]
                matrix[idx, :] = 0.0
                matrix[idx, idx] = 1.0
        for i in range(matrix.shape[0]):
            row_sum = matrix[i].sum()
            if row_sum == 0:
                matrix[i, i] = 1.0
            else:
                matrix[i] = matrix[i] / row_sum
        return matrix

    def _get_parent_matrix(self, segment_key: Tuple, mob: int, used_level: str) -> Optional[np.ndarray]:
        for parent_level in self.smoothing_cfg.get("parent_order", []):
            if parent_level == used_level:
                continue
            if parent_level == "segment_all":
                stats = self.segment_all_stats.get(segment_key)
            elif parent_level == "global_mob":
                stats = self.global_mob_stats.get((mob,))
            elif parent_level == "global_all":
                stats = self.global_all_stats.get(tuple())
            else:
                stats = None
            if stats and stats.ead_sum > 0 and stats.n_obs > 0:
                return self._build_probability_matrix(stats, parent_matrix=None)
        return None

    def select_matrix(self, segment_key: Tuple, mob: int) -> Tuple[np.ndarray, Dict]:
        stats, level, mob_used = self._select_candidate(segment_key, mob)
        meta = {
            "matrix_source": level,
            "mob_used": mob_used,
            "n_obs_used": 0,
            "ead_sum_used": 0.0,
        }
        if stats is None:
            matrix = np.eye(len(self.state_order))
            return matrix, meta

        parent_matrix = self._get_parent_matrix(segment_key, mob_used or mob, level)
        matrix = self._build_probability_matrix(stats, parent_matrix=parent_matrix)
        meta.update({"n_obs_used": stats.n_obs, "ead_sum_used": stats.ead_sum})
        return matrix, meta

    def segment_key(self, row: pd.Series) -> Tuple:
        return tuple(row[col] for col in self.segment_cols)
