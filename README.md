# CatSeek R1 · real BitNet b1.58

MiniAGI-style local agent backed by **Microsoft BitNet** (`microsoft/BitNet` / bitnet.cpp), not OpenAI.

## Exact BitNet stack

Same entrypoint flags as [`run_inference.py`](https://github.com/microsoft/BitNet):

| Flag | Meaning |
|------|---------|
| `-m` / `--model` | GGUF path |
| `-n` / `--n-predict` | Tokens to generate |
| `-p` / `--prompt` | Prompt (system prompt when `-cnv`) |
| `-t` / `--threads` | CPU threads |
| `-c` / `--ctx-size` | Context size |
| `-temp` / `--temperature` | Sampling temperature |
| `-cnv` / `--conversation` | Chat mode |

Preferred backend: `BITNET_HOME/run_inference.py` → `build/bin/llama-cli` with W1.58A8 kernels. HF transformers is an optional fallback.

## Setup

```bash
python3 "#catseeekr110.1.26.py" --setup-bitnet
python3 "#catseeekr110.1.26.py" --bitnet-status
```

## Usage

```bash
# BitNet one-shot (same as microsoft/BitNet)
python3 "#catseeekr110.1.26.py" -p "You are a helpful assistant" -cnv

# MiniAGI GUI
python3 "#catseeekr110.1.26.py"
python3 "#catseeekr110.1.26.py" "write hello cat in c"
```

## Env

See `.env_example` for `BITNET_HOME`, `BITNET_GGUF`, `BITNET_BACKEND`, threads/temp/n_predict, and MiniAGI knobs.
