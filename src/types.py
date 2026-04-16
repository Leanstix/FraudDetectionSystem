from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DatasetPaths:
    input_path: str
    dataset_root: str
    dataset_name: str
    transactions_path: str
    users_path: str
    locations_path: str
    sms_path: str
    mails_path: str


@dataclass(slots=True)
class TransactionRecord:
    transaction_id: str
    sender_id: str | None
    recipient_id: str | None
    transaction_type: str | None
    amount: float | None
    location: str | None
    payment_method: str | None
    sender_iban: str | None
    recipient_iban: str | None
    balance_after: float | None
    description: str | None
    timestamp: str | None


@dataclass(slots=True)
class UserRecord:
    first_name: str | None
    last_name: str | None
    birth_year: int | None
    salary: float | None
    job: str | None
    iban: str | None
    residence_city: str | None
    residence_lat: float | None
    residence_lng: float | None
    description: str | None


@dataclass(slots=True)
class LocationRecord:
    biotag: str | None
    timestamp: str | None
    lat: float | None
    lng: float | None
    city: str | None


@dataclass(slots=True)
class SmsRecord:
    thread_id: str
    sender: str | None
    recipient: str | None
    timestamp: str | None
    message_text: str
    raw_text: str


@dataclass(slots=True)
class MailRecord:
    thread_id: str
    sender: str | None
    recipient: str | None
    subject: str | None
    timestamp: str | None
    body_text: str
    raw_text: str


@dataclass(slots=True)
class AgentScore:
    transaction_id: str
    agent_name: str
    score: float
    reason: str


@dataclass(slots=True)
class SubmissionResult:
    output_path: str
    flagged_count: int
    total_transactions: int
    flagged_percentage: float
    transaction_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineArtifacts:
    dataset_name: str
    transactions_df: Any
    features_df: Any
    final_scores_df: Any
    submission: SubmissionResult
    diagnostics_path: str
    tracing_enabled: bool
    session_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
