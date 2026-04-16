# Reply Mirror Fraud Detection Agent System

This repository provides a complete, runnable, end-to-end Python solution for the Reply Mirror AI Agent Challenge.

It builds an adaptive, agent-based fraud detection pipeline that:
- ingests and normalizes challenge datasets
- resolves entities across transactions, users, locations, sms, and mails
- computes behavioral, temporal, geospatial, communication, and novelty signals
- fuses specialist agent scores into a final risk score
- writes a valid ASCII submission file (one transaction_id per line)
- writes diagnostics CSV for analysis and tuning

## Why This Satisfies The Agent-Based Requirement

The system is explicitly composed of cooperative specialist agents, each with a focused role:
- DataIngestionAgent: load + normalize
- EntityResolutionAgent: heuristic user/location/communication linking
- TransactionBehaviorAgent: amount/profile behavior anomalies
- TemporalSequenceAgent: sequence and burst anomalies
- GeoSpatialAgent: location inconsistency and novelty
- CommunicationRiskAgent: thread-level communication risk (heuristics first, optional LLM)
- NoveltyDriftAgent: unsupervised novelty/drift scoring
- FusionDecisionAgent: deterministic score fusion + thresholding
- SubmissionWriter: strict output validation + ASCII writing

This is adaptive and non-static because multiple weak signals, novelty detection, and communication analysis are combined instead of hardcoded rules or single deterministic logic.

## Architecture (Text Diagram)

1. Load + normalize data
2. Resolve entities and contextual links
3. Build feature store
4. Run specialist scoring agents in parallel style
5. Fuse scores and choose deterministic threshold with bounds
6. Validate and write submission
7. Write diagnostics CSV
8. Flush tracing

## Tracing and OpenRouter Integration

Tracing follows the canonical pattern from track-your-submission:
- python-dotenv for env loading
- ChatOpenAI via OpenRouter base URL: https://openrouter.ai/api/v1
- Langfuse client + CallbackHandler
- session id format: {TEAM_NAME}-{ULID}
- metadata includes: {"langfuse_session_id": session_id, "dataset_name": ..., "task_name": ...}
- flush called at pipeline end

Notes:
- Langfuse SDK is pinned to v3-compatible range.
- Secrets are never printed.
- If tracing keys are missing, execution continues safely.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

Inspect dataset:

```bash
python -m src.main inspect --input "Brave New World - train.zip"
```

Create submission:

```bash
python -m src.main predict --input "Brave New World - train.zip" --output "outputs/brave_new_world_submission.txt"
```

Verbose mode:

```bash
python -m src.main predict --input "Brave New World - train.zip" --output "outputs/brave_new_world_submission.txt" --verbose
```

No LLM mode:

```bash
python -m src.main predict --input "Brave New World - train.zip" --output "outputs/brave_new_world_submission.txt" --no-llm
```

If your zip is under hackTheCode, this also works:

```bash
python -m src.main predict --input "hackTheCode/Brave+New+World+-+train.zip" --output "outputs/brave_new_world_submission.txt"
```

## Output Files

- Submission: outputs/brave_new_world_submission.txt
  - ASCII text only
  - one transaction_id per line
  - deterministic ordering: final_risk_score desc, transaction_id asc

- Diagnostics: outputs/brave_new_world_diagnostics.csv
  - per-transaction scores from all agents
  - final risk score
  - threshold metadata
  - top risk reasons

## Assumptions and Limitations

- No fraud labels are assumed in training data (unsupervised/weakly-supervised approach).
- Entity matching is heuristic and probabilistic due imperfect identifiers.
- Communication LLM analysis is selective and cached; heuristic fallback is always available.
- Geospatial distances use inferred city centroids and linked GPS points where available.

## Next Tuning Steps (Future Challenge Levels)

1. Learn dynamic calibration for target flag-rate by dataset difficulty profile.
2. Improve entity graph linking with probabilistic record linkage over multilingual text.
3. Add drift monitoring across rolling windows and domain adaptation per level.
4. Add richer communication-to-transaction causality linking using event graphs.
5. Introduce online-learning feedback loops from post-evaluation diagnostics.
