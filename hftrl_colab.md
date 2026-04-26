# MedFlow HF-TRL Training (Colab)

## 1) Install dependencies

```bash
pip install -q openenv-core==0.2.3 transformers datasets accelerate trl matplotlib
```

## 2) Clone repository

```bash
git clone <your-repo-url>
cd MedFlow-OpenEnv
```

## 3) Run training

```bash
python -m train.train_hftrl --model sshleifer/tiny-gpt2 --epochs 1 --outdir outputs/evals
```

## 4) Generated artifacts

- `outputs/evals/trl_loss_curve.png`
- `outputs/evals/trl_reward_curve.png`
- `outputs/evals/trl_training_metrics.csv`
- `outputs/evals/trl_summary.json`
