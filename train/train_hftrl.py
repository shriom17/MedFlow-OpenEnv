"""Colab-friendly HF-TRL / Unsloth demo generator for MedFlow-OpenEnv.

This script writes a copy-pasteable Colab snippet (markdown) that shows how to
install dependencies and run a minimal PPO-style training loop with HF TRL,
using the `app.openenv_medflow` environment as the reward source.

Run locally to generate `hftrl_colab.md`, then paste into a Colab cell.
"""
import argparse
import textwrap
from pathlib import Path


COLAB_MD = """
# HF-TRL minimal demo for MedFlow-OpenEnv

## 1) Install dependencies
Run this in a Colab cell:

```bash
pip install -q trl transformers accelerate datasets safetensors
```

## 2) Mount repo / upload files
Either clone your repo in Colab or upload the repository files so `app/openenv_medflow.py` is available.

```bash
git clone <your-repo-url>
cd MedFlow-OpenEnv
```

## 3) Minimal HF-TRL training loop (copy into a Python cell)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
import importlib
import random

# Load environment from the repository
env_mod = importlib.import_module('app.openenv_medflow')
env = env_mod.make_env()

# Load a small model for demo (use 'gpt2' or a smaller distilgpt2 for faster runs)
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# PPO config (very small for demo)
ppo_config = PPOConfig(batch_size=1, forward_batch_size=1)
trainer = PPOTrainer(model=model, tokenizer=tokenizer, **ppo_config.__dict__)

def reward_from_response(response_text):
    # Convert generated text to an environment action and query env for reward
    # NOTE: This is a placeholder: adapt parsing to map model outputs -> actions
    action = response_text.strip().split('\n')[-1]
    _, reward, done, _ = env.step(action)
    return float(reward)

prompts = ["Patient: chest pain. Next step:"]

for epoch in range(10):
    for prompt in prompts:
        query_tensors = tokenizer(prompt, return_tensors='pt')
        # generate sample and compute reward
        response = trainer.generate(query_tensors['input_ids'], max_length=64)
        text = tokenizer.decode(response[0], skip_special_tokens=True)
        r = reward_from_response(text)
        # Perform PPO step
        trainer.step([prompt], [text], rewards=[r])
    print(f"Epoch {epoch} done")

```

Notes:
- This demo omits many production concerns (reward shaping, batching, dataset management,
  tokenization alignment, and safety). Use it as a starting template.

"""


def main(output: Path):
    output.write_text(textwrap.dedent(COLAB_MD))
    print(f"Wrote Colab demo to {output}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('hftrl_colab.md'))
    args = p.parse_args()
    main(args.out)
