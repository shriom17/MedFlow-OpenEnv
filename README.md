# MedFlow-OpenEnv

MedFlow-OpenEnv is a real-world OpenEnv environment that simulates hospital queue triage and resource allocation. An agent must prioritize patients, assign doctors, manage limited beds, and handle dynamic arrivals across easy, medium, and hard scenarios.

## Why This Environment

This environment models a real operational workflow used in healthcare systems:

- triage under uncertainty
- constrained resource scheduling (beds + specialists)
- safety-critical prioritization (emergency response)
- quality-of-care tradeoffs (speed vs specialization match)

## OpenEnv Interface

The environment is exposed via the OpenEnv HTTP server created in `app/main.py`.

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /metadata`
- `GET /health`

Core implementation files:

- `app/models.py`: typed action/observation/state models
- `app/env.py`: environment dynamics, rewards, episode logic
- `app/tasks.py`: task catalog + deterministic graders
- `openenv.yaml`: environment metadata and task definitions

## Action Space

`HospitalAction`

- `action_type`: `assign | prioritize | discharge | wait`
- `patient_id`: optional `int` (required for assign/prioritize/discharge)
- `doctor_id`: optional `int` (required for assign)

## Observation Space

`HospitalObservation`

- queue snapshot (`waiting_patients`, `queue_length`)
- doctor status (`doctors` with busy/free details)
- capacity state (`beds_available`)
- time/progress (`current_time_minutes`, `progress_score`)
- RL signals (`reward`, `done`, `step_feedback`)

## Tasks (Easy → Medium → Hard)

1. `easy_small_clinic`
- 5 patients, 2 general doctors, 4 beds
- objective: clear queue quickly and treat emergencies first

2. `medium_busy_opd`
- 10 initial patients + 3 dynamic arrivals
- mixed specialists + routing quality pressure

3. `hard_mass_casualty`
- 8 initial critical patients + 10 wave arrivals
- critical death penalty for high-severity untreated wait

All graders produce deterministic scores in `[0.0, 1.0]`.

## Reward Design

The environment uses shaped rewards plus an end-of-episode task grader score:

- positive: timely emergency/urgent handling, normal throughput, discharge
- negative: specialization mismatch, emergency neglect, invalid/overflow behavior
- final: grader-based normalized task score `0.0..1.0`

This gives dense trajectory feedback and clear terminal outcomes.

## Local Setup

### 1) Create environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Run tests

```powershell
pytest -q
```

### 3) Start API server

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Baseline Inference

### A) Greedy baseline (local deterministic policy)

```powershell
python -m app.baseline --seed 42
```

Runs all 3 tasks and prints per-task total reward.

### B) OpenAI baseline (model-driven)

```powershell
$env:OPENAI_API_KEY="<your_key>"
$env:OPENAI_MODEL="gpt-4.1-mini"
python -m app.baseline_openai --seed 42
```

This script evaluates all tasks and prints per-task JSON results plus average final score.

## Docker

Build and run:

```powershell
docker build -t medflow-openenv .
docker run --rm -p 7860:7860 medflow-openenv
```

Container entrypoint runs:

```text
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## OpenEnv Validation

After installing OpenEnv CLI tooling, run:

```powershell
openenv validate
```

If your CLI expects a file argument, use:

```powershell
openenv validate openenv.yaml
```

## Hugging Face Spaces Deployment

- Use this repo with Docker Space configuration
- Ensure Space is tagged with `openenv`
- Exposed port is `7860`
- Entry app is `app.main:app`

## Project Structure

```text
app/
	main.py             # OpenEnv HTTP app bootstrap
	env.py              # Environment simulation logic
	models.py           # Typed action/observation/state models
	tasks.py            # Tasks + graders
	baseline.py         # Greedy deterministic baseline
	baseline_openai.py  # OpenAI API baseline evaluator
tests/
	test_environment.py
openenv.yaml
Dockerfile
requirements.txt
```
