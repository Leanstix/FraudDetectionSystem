from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.pipeline.orchestrator import FraudPipeline


def _dataset_zip() -> str:
    candidates = [
        Path("Brave New World - train.zip"),
        Path("hackTheCode/Brave New World - train.zip"),
        Path("hackTheCode/Brave+New+World+-+train.zip"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = list(Path(".").glob("**/Brave*train.zip"))
    if not found:
        raise FileNotFoundError("Brave New World train zip not found")
    return str(found[0])


def test_pipeline_end_to_end_creates_outputs(tmp_path: Path):
    output_path = tmp_path / "brave_new_world_submission.txt"
    settings = Settings.from_env_and_file()

    pipeline = FraudPipeline(
        settings=settings,
        input_path=_dataset_zip(),
        output_path=str(output_path),
        dataset_name="brave_new_world",
        no_llm=True,
        verbose=False,
    )
    artifacts = pipeline.run()

    assert output_path.exists()
    assert Path(artifacts.diagnostics_path).exists()
    assert artifacts.submission.flagged_count > 0
    assert artifacts.submission.flagged_count < artifacts.submission.total_transactions
