from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent


class TransactionBehaviorAgent(BaseAgent):
    name = "transaction_behavior"

    def run(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df = features_df.copy()

        z_sender = df.get("amount_robust_z_sender", 0).astype(float).abs().clip(0, 6) / 6.0
        z_recipient = df.get("amount_robust_z_recipient", 0).astype(float).abs().clip(0, 6) / 6.0
        new_pair = df.get("new_sender_recipient_pair", 0).astype(float)
        new_method = df.get("new_payment_method_for_sender", 0).astype(float)
        new_type = df.get("new_transaction_type_for_sender", 0).astype(float)
        novelty = df.get("novelty_score", 0).astype(float)

        score = (
            0.30 * z_sender
            + 0.20 * z_recipient
            + 0.15 * new_pair
            + 0.12 * new_method
            + 0.10 * new_type
            + 0.13 * novelty
        ).clip(0.0, 1.0)

        reasons = []
        for i in range(len(df)):
            tokens: list[str] = []
            if z_sender.iloc[i] > 0.65:
                tokens.append("sender_amount_outlier")
            if z_recipient.iloc[i] > 0.65:
                tokens.append("recipient_amount_outlier")
            if new_pair.iloc[i] > 0:
                tokens.append("new_counterparty_pair")
            if new_method.iloc[i] > 0:
                tokens.append("new_payment_method")
            if new_type.iloc[i] > 0:
                tokens.append("new_transaction_type")
            if novelty.iloc[i] > 0.7:
                tokens.append("high_novelty")
            reasons.append(";".join(tokens[:4]) if tokens else "baseline_behavior")

        return pd.DataFrame(
            {
                "transaction_id": df["transaction_id"],
                "transaction_behavior_score": score,
                "transaction_behavior_reason": reasons,
            }
        )
