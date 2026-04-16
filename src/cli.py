from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.pipeline.orchestrator import FraudPipeline


def infer_dataset_name(input_path: str) -> str:
    base = Path(input_path).stem.lower().replace(" ", "_").replace("+", "_")
    base = base.replace("-", "_")
    return "_".join([token for token in base.split("_") if token])


def run_inspect(input_path: str, config_path: str | None = None) -> dict:
    settings = Settings.from_env_and_file(config_path)
    dataset_name = infer_dataset_name(input_path)
    pipeline = FraudPipeline(
        settings=settings,
        input_path=input_path,
        output_path="outputs/inspect_only.txt",
        dataset_name=dataset_name,
        no_llm=True,
        verbose=False,
    )
    report = pipeline.inspect()
    print(f"dataset name: {report['dataset_name']}")
    print(f"total transaction count: {report['transactions']}")
    print(f"users: {report['users']}")
    print(f"locations: {report['locations']}")
    print(f"sms threads: {report['sms_threads']}")
    print(f"mail threads: {report['mail_threads']}")
    return report


def run_predict(input_path: str, output_path: str, no_llm: bool = False, verbose: bool = False, config_path: str | None = None):
    settings = Settings.from_env_and_file(config_path)
    dataset_name = infer_dataset_name(input_path)
    pipeline = FraudPipeline(
        settings=settings,
        input_path=input_path,
        output_path=output_path,
        dataset_name=dataset_name,
        no_llm=no_llm,
        verbose=verbose,
    )
    artifacts = pipeline.run()
    pipeline.print_summary(artifacts)
    return artifacts
