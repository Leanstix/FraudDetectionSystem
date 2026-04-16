from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent
from src.config import Settings
from src.models.calibration import choose_target_count, threshold_from_target
from src.models.fusion import weighted_fusion


class FusionDecisionAgent(BaseAgent):
    name = "fusion_decision"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._target_count = 1

    def merge_scores(self, features_df: pd.DataFrame, agent_outputs: list[pd.DataFrame]) -> pd.DataFrame:
        merged = features_df.copy()
        for out in agent_outputs:
            merged = merged.merge(out, on="transaction_id", how="left")
        return merged

    def compute_final_score(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        df = merged_df.copy()
        score_cols = [
            "transaction_behavior_score",
            "temporal_sequence_score",
            "geospatial_score",
            "communication_risk_score",
            "novelty_drift_score",
        ]
        for col in score_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0).astype(float).clip(0, 1)

        df["final_risk_score"] = weighted_fusion(df, self.settings.agent_weights).clip(0, 1)

        reason_pairs = [
            ("transaction_behavior_score", "transaction_behavior_reason"),
            ("temporal_sequence_score", "temporal_sequence_reason"),
            ("geospatial_score", "geospatial_reason"),
            ("communication_risk_score", "communication_risk_reason"),
            ("novelty_drift_score", "novelty_drift_reason"),
        ]
        top_reasons: list[str] = []
        for _, row in df.iterrows():
            candidates: list[tuple[float, str]] = []
            for score_col, reason_col in reason_pairs:
                reason = str(row.get(reason_col, "")).strip()
                score = float(row.get(score_col, 0.0))
                if reason:
                    candidates.append((score, reason))
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_reasons.append(" | ".join(reason for _, reason in candidates[:3]))
        df["top_risk_reasons"] = top_reasons
        return df

    def choose_threshold(self, scored_df: pd.DataFrame) -> float:
        n = len(scored_df)
        th = self.settings.thresholds
        self._target_count = choose_target_count(
            n,
            target_rate=float(th.get("target_flag_rate", 0.12)),
            min_rate=float(th.get("min_flag_rate", 0.02)),
            max_rate=float(th.get("max_flag_rate", 0.45)),
        )
        threshold = threshold_from_target(scored_df["final_risk_score"], self._target_count)
        return float(np.clip(threshold, float(th.get("min_score", 0.0)), float(th.get("max_score", 1.0))))

    def flag_transactions(self, scored_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        df = scored_df.copy().sort_values(["final_risk_score", "transaction_id"], ascending=[False, True]).reset_index(drop=True)
        df["flagged"] = False

        target = int(np.clip(self._target_count, 1, max(len(df) - 1, 1)))
        df.loc[: target - 1, "flagged"] = True

        # Preserve threshold information for diagnostics while enforcing valid output cardinality.
        df["threshold_used"] = threshold
        return df

    def run(self, features_df: pd.DataFrame, agent_outputs: list[pd.DataFrame]) -> pd.DataFrame:
        merged = self.merge_scores(features_df, agent_outputs)
        scored = self.compute_final_score(merged)
        threshold = self.choose_threshold(scored)
        flagged = self.flag_transactions(scored, threshold)
        return flagged[["transaction_id", "final_risk_score", "flagged", "top_risk_reasons"]]
