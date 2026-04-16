from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agents.communication_risk_agent import CommunicationRiskAgent
from src.agents.entity_resolution_agent import EntityResolutionAgent
from src.agents.fusion_decision_agent import FusionDecisionAgent
from src.agents.geospatial_agent import GeoSpatialAgent
from src.agents.ingestion_agent import DataIngestionAgent
from src.agents.novelty_drift_agent import NoveltyDriftAgent
from src.agents.submission_writer import SubmissionWriter
from src.agents.temporal_sequence_agent import TemporalSequenceAgent
from src.agents.transaction_behavior_agent import TransactionBehaviorAgent
from src.config import Settings
from src.data.dataset_inspector import DatasetInspector
from src.data.entity_resolution import EntityResolver
from src.data.feature_store import FeatureStore
from src.data.loaders import DatasetLoader
from src.data.normalize import Normalizer
from src.llm.client import LLMClient
from src.llm.communication_analyzer import CommunicationAnalyzer
from src.tracing import TracingManager
from src.types import PipelineArtifacts
from src.utils.io import write_csv


class FraudPipeline:
    def __init__(self, settings: Settings, input_path: str, output_path: str, dataset_name: str, no_llm: bool = False, verbose: bool = False):
        self.settings = settings
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.no_llm = no_llm
        self.verbose = verbose

        if no_llm:
            self.settings.llm_enabled = False

        self.loader = DatasetLoader(input_path)
        self.normalizer = Normalizer()
        self.ingestion_agent = DataIngestionAgent(self.loader, self.normalizer)

        self.resolver = EntityResolver()
        self.entity_agent = EntityResolutionAgent(self.resolver)
        self.feature_store = FeatureStore()

        self.tracing = TracingManager(settings)
        self.llm_client = LLMClient(settings, self.tracing)
        self.communication_analyzer = CommunicationAnalyzer(self.llm_client)

        self.transaction_agent = TransactionBehaviorAgent()
        self.temporal_agent = TemporalSequenceAgent()
        self.geo_agent = GeoSpatialAgent()
        self.communication_agent = CommunicationRiskAgent(self.communication_analyzer)
        self.novelty_agent = NoveltyDriftAgent()

        self.fusion_agent = FusionDecisionAgent(settings)
        self.submission_writer = SubmissionWriter()

    def inspect(self) -> dict:
        return DatasetInspector(self.input_path).inspect()

    def run(self) -> PipelineArtifacts:
        session_id = None
        if self.tracing.is_enabled():
            session_id = self.tracing.generate_session_id()

        # 1. load + normalize
        data = self.ingestion_agent.run()

        # 2. entity resolution
        entity_out = self.entity_agent.run(
            transactions_df=data["transactions"],
            users_df=data["users"],
            locations_df=data["locations"],
            sms_df=data["sms"],
            mails_df=data["mails"],
        )
        enriched_transactions = entity_out["enriched_transactions_df"]

        # 3. feature building
        features_df = self.feature_store.build_all(
            transactions_df=enriched_transactions,
            users_df=data["users"],
            locations_df=data["locations"],
            sms_df=data["sms"],
            mails_df=data["mails"],
        )

        # 4. run all specialist agents
        transaction_scores = self.transaction_agent.run(features_df)
        temporal_scores = self.temporal_agent.run(features_df)
        geo_scores = self.geo_agent.run(features_df)
        comm_scores = self.communication_agent.run(features_df, data["sms"], data["mails"], self.dataset_name)
        novelty_scores = self.novelty_agent.run(features_df)

        outputs = [transaction_scores, temporal_scores, geo_scores, comm_scores, novelty_scores]

        # 5. fuse scores
        merged = self.fusion_agent.merge_scores(features_df, outputs)
        scored = self.fusion_agent.compute_final_score(merged)
        threshold = self.fusion_agent.choose_threshold(scored)
        flagged_full = self.fusion_agent.flag_transactions(scored, threshold)

        # 6. validate + write submission
        submission_input = flagged_full[["transaction_id", "final_risk_score", "flagged", "top_risk_reasons"]].copy()
        submission = self.submission_writer.run(submission_input, enriched_transactions, self.output_path)

        # 7. write diagnostics file
        diagnostics_path = str(Path(self.output_path).with_name(Path(self.output_path).stem.replace("_submission", "") + "_diagnostics.csv"))
        diagnostics_df = flagged_full.sort_values(["final_risk_score", "transaction_id"], ascending=[False, True]).copy()
        write_csv(diagnostics_path, diagnostics_df)

        # 8. flush tracing
        self.tracing.flush()

        artifacts = PipelineArtifacts(
            dataset_name=self.dataset_name,
            transactions_df=enriched_transactions,
            features_df=features_df,
            final_scores_df=flagged_full,
            submission=submission,
            diagnostics_path=diagnostics_path,
            tracing_enabled=self.tracing.is_enabled(),
            session_id=session_id,
            metadata={
                "entity_resolution": entity_out.get("linking_diagnostics", {}),
                "threshold": threshold,
            },
        )
        return artifacts

    def print_summary(self, artifacts: PipelineArtifacts) -> None:
        total = artifacts.submission.total_transactions
        flagged = artifacts.submission.flagged_count
        pct = (100.0 * flagged / total) if total else 0.0

        print(f"dataset name: {artifacts.dataset_name}")
        print(f"total transaction count: {total}")
        print(f"total flagged count: {flagged}")
        print(f"flagged percentage: {pct:.2f}%")
        print(f"tracing enabled: {artifacts.tracing_enabled}")
        if artifacts.tracing_enabled and artifacts.session_id:
            print(f"generated session ID: {artifacts.session_id}")
        print(f"output submission path: {artifacts.submission.output_path}")
        print(f"diagnostics path: {artifacts.diagnostics_path}")
