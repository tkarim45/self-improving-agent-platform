# 03 — Setup (Apple M1, 8 GB)

## 0. Golden rules for this machine

- **Never** use conda `base` or system Python (enforced by a hook). Use the `personal`
  env for this project, `claude` for throwaway scratch.
- Usable model memory ≈ 4–5 GB after the OS. Keep one large model resident at a time.
- Cloud creds live in the global `~/.env` — source it, never paste secrets, never commit.

## 1. Python environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate personal
# or absolute path in one-shot shells:
# ~/miniconda3/envs/personal/bin/python
pip install -r requirements.txt
```

Core deps (`requirements.txt`): `fastapi uvicorn pydantic anthropic[bedrock]
sentence-transformers rank-bm25 faiss-cpu networkx redis mcp mlx mlx-lm
llama-cpp-python duckdb pytest ruff black python-dotenv streamlit prometheus-client`.

## 2. Local models (llama.cpp, Metal)

Install with Metal support (already proven on this machine by `llm-quantization-bench`):

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

Download a small GGUF for the cheap tier (pick ONE to start; 1.5B Q4 ≈ 1 GB):

```bash
# via huggingface-hub
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF','qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='models')"
```

Sanity check it runs on Metal:

```bash
python - <<'PY'
from llama_cpp import Llama
llm = Llama(model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf", n_gpu_layers=-1, n_ctx=2048)
print(llm("Q: capital of France? A:", max_tokens=8)["choices"][0]["text"])
PY
```

Model size guide for 8 GB (leave headroom):
| Model | Q4 size | Fits? |
|---|---|---|
| Qwen2.5-0.5B | ~0.4 GB | ✅ easy |
| Qwen2.5-1.5B / Llama-3.2-1B | ~1 GB | ✅ comfortable |
| Llama-3.2-3B | ~2 GB | ⚠️ tight |
| 7B | ~4.5 GB | ⚠️ swaps — avoid on this machine |

## 3. On-device fine-tuning (MLX — for the flywheel, Milestone 5)

Use **MLX**, not PyTorch (unified-memory efficient on M-series).

```bash
pip install mlx mlx-lm

# LoRA fine-tune a small base on your (query,good,bad) or instruction data
python -m mlx_lm.lora \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train \
  --data ./data/lora \        # expects train.jsonl / valid.jsonl
  --batch-size 1 \            # keep small on 8 GB
  --num-layers 8 \            # tune fewer layers to save memory
  --iters 400 \
  --adapter-path ./adapters/reranker

# fuse adapter into a standalone model when promoting a candidate
python -m mlx_lm.fuse --model Qwen/Qwen2.5-1.5B-Instruct --adapter-path ./adapters/reranker
```

Memory tips: `--batch-size 1`, low `--num-layers`, 4-bit base (`--model` a `-4bit` MLX
repo), close other apps. If it OOMs, drop to a 0.5B base or fewer layers.

## 4. Cloud creds (Claude / Bedrock)

```bash
set -a; source ~/.env; set +a   # loads ANTHROPIC_API_KEY / AWS_* etc.
```

Check `~/.env.example` for which keys exist. If a needed key is missing, add the empty
key to both `~/.env` and `~/.env.example` and ask — do not invent values.

## 5. First run (once Milestone 0+ exists)

```bash
make dev        # API + workers
make eval       # run golden eval set
make test       # pytest
```

## 6. Troubleshooting

- **OOM during inference** → smaller model / lower `n_ctx` / fewer `n_gpu_layers`.
- **OOM during MLX LoRA** → `--batch-size 1`, fewer `--num-layers`, 0.5B base.
- **Metal not used** → reinstall `llama-cpp-python` with `CMAKE_ARGS="-DGGML_METAL=on"`.
- **Slow** → this is an 8 GB M1; prefer 0.5–1.5B, batch small, cache aggressively.
