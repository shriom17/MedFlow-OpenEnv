---
title: MedFlow OpenEnv
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: Hospital queue management simulation API for AI agents
---

# Hospital Queue Management Environment (OpenEnv)

Hospital Queue Management Environment for AI agents using the OpenEnv specification.

## Purpose

This environment simulates real-world hospital triage and resource allocation. Agents must balance urgency, doctor availability, specialization matching, and bed constraints.

## Space Usage

Base URL (replace with your live Space URL):

- https://<your-space>.hf.space

Useful endpoints:

- GET /health
- GET /docs
- GET /schema
- POST /reset
- POST /step
- GET /state

Swagger UI:

- https://<your-space>.hf.space/docs

## Quick API Example (curl)

Reset:

```bash
curl -X POST "https://<your-space>.hf.space/reset" \
	-H "Content-Type: application/json" \
	-d "{\"task_id\":\"easy_small_clinic\",\"seed\":42}"
```

Step:

```bash
curl -X POST "https://<your-space>.hf.space/step" \
	-H "Content-Type: application/json" \
	-d "{\"action\":{\"action_type\":\"wait\"}}"
```

Get state:

```bash
curl "https://<your-space>.hf.space/state"
```

## Postman Example

1. Create request: POST https://<your-space>.hf.space/reset
2. Body (JSON): {"task_id":"easy_small_clinic","seed":42}
3. Create request: POST https://<your-space>.hf.space/step
4. Body (JSON): {"action":{"action_type":"assign","patient_id":1,"doctor_id":1}}
5. Create request: GET https://<your-space>.hf.space/state

## Tasks Supported

- Easy: small clinic (`easy_small_clinic`)
- Medium: busy OPD (`medium_busy_opd`)
- Hard: mass casualty (`hard_mass_casualty`)

## Reward and Grader (High-Level)

Step reward is shaped for partial progress:

- Positive for timely assignment/discharge and emergency handling
- Negative for invalid actions, specialization mismatch, and neglected emergencies

At episode end, each task uses a deterministic grader that returns a normalized score in [0.0, 1.0].

## Baseline Testing

Greedy baseline (local deterministic policy):

```bash
python -m app.baseline --seed 42
```

OpenAI baseline (if API key is configured):

```bash
export OPENAI_API_KEY=<your_key>
python -m app.baseline_openai --seed 42
```

Submission inference entrypoint (required file name):

```bash
export API_BASE_URL=<your_model_api_base>
export MODEL_NAME=<your_model_name>
export HF_TOKEN=<your_api_token>
python inference.py --seed 42
```

`inference.py` uses the OpenAI client internally and reads the required submission variables.

### Sample Baseline Scores (Seed=42)

| Task | Baseline type | Final grader score (0.0-1.0) |
|---|---|---|
| easy_small_clinic | greedy | 0.20 |
| medium_busy_opd | greedy | 0.40 |
| hard_mass_casualty | greedy | 0.25 |

## Local Validation

Run tests:

```bash
pytest -q
```

Run OpenEnv validator:

```bash
openenv validate .
```
