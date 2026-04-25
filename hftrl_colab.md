
# HF-TRL minimal demo for MedFlow-OpenEnv

## 1) Install dependencies
Run this in a Colab cell:

```bash
pip install -q trl transformers accelerate datasets safetensors
```

## 2) Mount repo / upload files
Either clone your repo in Colab or upload the repository files so `app/env.py` and `app/models.py` are available.

```bash
git clone <your-repo-url>
cd MedFlow-OpenEnv
```

## 3) Minimal HF-TRL training loop (copy into a Python cell)

```python
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer

from app.env import HospitalQueueEnvironment
from app.models import HospitalAction


def parse_action(text):
    '''Extract a JSON action object from model output.'''
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return HospitalAction(action_type="wait")

    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return HospitalAction(action_type="wait")

    action_type = str(payload.get("action_type", "wait")).lower()
    if action_type not in {"assign", "prioritize", "discharge", "wait"}:
        action_type = "wait"

    def to_int(value):
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return HospitalAction(
        action_type=action_type,
        patient_id=to_int(payload.get("patient_id")),
        doctor_id=to_int(payload.get("doctor_id")),
    )


# Load the actual structured environment from the repo.
env = HospitalQueueEnvironment()
obs = env.reset(task_id="easy_small_clinic")

# Load a small model for demo (use 'gpt2' or distilgpt2 for faster runs).
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# PPO config (very small for demo). TRL APIs vary slightly by version, so keep
# this cell as a template rather than a drop-in production trainer.
ppo_config = PPOConfig(batch_size=1, forward_batch_size=1)
trainer = PPOTrainer(model=model, tokenizer=tokenizer, **ppo_config.__dict__)

prompts = [
    "Patient queue is waiting. Return JSON with action_type, patient_id, doctor_id."
]

for epoch in range(10):
    for prompt in prompts:
        query_tensors = tokenizer(prompt, return_tensors="pt")
        response_tensors = trainer.generate(query_tensors["input_ids"], max_new_tokens=64)
        response_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
        action = parse_action(response_text)
        obs = env.step(action)
        reward = float(obs.reward or 0.0)
        trainer.step([query_tensors["input_ids"][0]], [response_tensors[0]], rewards=[reward])
    print(f"Epoch {epoch} done")

```

Notes:
- This demo omits many production concerns (reward shaping, batching, dataset management,
  tokenization alignment, and safety). Use it as a starting template.
- The environment is now the structured hospital queue simulator in `app/env.py`,
  not the earlier toy scaffold.

