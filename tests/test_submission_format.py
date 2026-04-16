from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agents.submission_writer import SubmissionWriter


def test_submission_output_format_and_ordering(tmp_path: Path):
    transactions_df = pd.DataFrame(
        {
            "transaction_id": ["a", "b", "c", "d"],
        }
    )
    flagged_df = pd.DataFrame(
        {
            "transaction_id": ["d", "b", "a", "c"],
            "final_risk_score": [0.6, 0.9, 0.9, 0.3],
            "flagged": [True, True, True, False],
            "top_risk_reasons": ["r1", "r2", "r3", "r4"],
        }
    )

    out = tmp_path / "submission.txt"
    writer = SubmissionWriter()
    result = writer.run(flagged_df=flagged_df, transactions_df=transactions_df, output_path=str(out))

    text = out.read_text(encoding="ascii")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    assert lines
    assert len(lines) < len(transactions_df)
    assert set(lines).issubset(set(transactions_df["transaction_id"]))
    text.encode("ascii")
    assert lines == ["a", "b", "d"]  # score desc then transaction_id asc
    assert result.flagged_count == 3
