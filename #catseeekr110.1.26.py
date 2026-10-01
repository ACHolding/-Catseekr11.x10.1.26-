#!/usr/bin/env python3
"""
CatSeek R1 / CatR1-MiniAGI — real BitNet b1.58 (no OpenAI)

Local MiniAGI-style agent powered by Microsoft BitNet b1.58 inference:
  1) bitnet.cpp (llama-cli) — preferred, real W1.58A8 kernels
  2) HuggingFace transformers BitNetForCausalLM — portable fallback
  3) heuristic stub only when no model weights / runtime are present

Setup once:
  python3 "#catseeekr110.1.26.py" --setup-bitnet

Usage:
  python3 "#catseeekr110.1.26.py"
  python3 "#catseeekr110.1.26.py" "write hello cat in c"
  python3 "#catseeekr110.1.26.py" --bitnet-status
"""

from __future__ import annotations

import json
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk

APP_NAME = "CatSeek R1 · BitNet MiniAGI [c] Kondo Solutions 1999-2026"
AGENT_NAME = "CatSeek R1"
BITNET_MODEL = "microsoft/bitnet-b1.58-2B-4T"
BITNET_SUMMARIZER = "microsoft/bitnet-b1.58-2B-4T"
BITNET_BITS = 1.58
BITNET_GGUF_REPO = "microsoft/BitNet-b1.58-2B-4T-gguf"
BITNET_GIT = "https://github.com/microsoft/BitNet.git"
NO_OPENAI = True
OPERATING_SYSTEM = platform.platform()
MEMORY_DIR = Path.home() / ".catr1-miniagi"
MEMORY_FILE = MEMORY_DIR / "memory.json"
_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ENV_FILE = _ROOT / ".env"
ENV_EXAMPLE = _ROOT / ".env_example"
DEFAULT_BITNET_HOME = Path.home() / "BitNet"
DEFAULT_BITNET_GGUF = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
MEMORY_VERSION = 1
MAX_LOG_ENTRIES = 500
MAX_STEPS_DEFAULT = 12

# MiniAGI command surface (muellerberndt/mini-agi).
MINIAGI_COMMANDS = (
    "memorize_thoughts",
    "execute_python",
    "execute_shell",
    "ingest_data",
    "process_data",
    "web_search",
    "talk_to_user",
    "done",
)
# Offline helpers (CatSeek R1 local coder — not in stock MiniAGI).
LOCAL_COMMANDS = ("write_code", "answer")
ALL_COMMANDS = MINIAGI_COMMANDS + LOCAL_COMMANDS


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + "\n…[truncated]"


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _parse_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# MiniAGI parameter pack (exact .env_example keys)
# ---------------------------------------------------------------------------

class MiniAGIParams:
    """MiniAGI-shaped parameters backed only by local BitNet (no OpenAI)."""

    DEFAULTS = {
        # Real BitNet b1.58 — no OpenAI keys or cloud chat APIs
        "MODEL": BITNET_MODEL,
        "SUMMARIZER_MODEL": BITNET_SUMMARIZER,
        "ENABLE_CRITIC": False,
        "PROMPT_USER": True,
        "MAX_CONTEXT_SIZE": 4000,
        "MAX_MEMORY_ITEM_SIZE": 2000,
        "SUMMARIZER_CHUNK_SIZE": 3000,
        "WORK_DIR": "",
        "DEBUG": False,
        "FILES_OFF": True,
        "MAX_STEPS": MAX_STEPS_DEFAULT,
        "REASONING_EFFORT": 64,
        "BITNET_BITS": BITNET_BITS,
        "BITNET_HOME": str(DEFAULT_BITNET_HOME),
        "BITNET_GGUF": DEFAULT_BITNET_GGUF,
        "BITNET_BACKEND": "auto",  # auto | cpp | hf | stub
        "BITNET_THREADS": max(1, (os.cpu_count() or 2) // 2),
        "BITNET_TEMP": 0.7,
        "BITNET_N_PREDICT": 256,
    }

    def __init__(self, values: dict | None = None):
        self.values = dict(self.DEFAULTS)
        if values:
            self.update(values)
        self.apply_work_dir()

    @classmethod
    def from_env_file(cls, path: Path = ENV_FILE) -> "MiniAGIParams":
        raw: dict = {}
        # Process env first, then .env overrides for empty keys only if not in os.environ
        for key in cls.DEFAULTS:
            if key in os.environ:
                raw[key] = os.environ[key]
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k == "OPENAI_API_KEY":
                    continue  # ignored — BitNet only
                if k in cls.DEFAULTS:
                    raw[k] = v
        raw.pop("OPENAI_API_KEY", None)
        return cls(raw)

    def update(self, values: dict):
        for k, v in values.items():
            if k == "OPENAI_API_KEY":
                continue
            if k not in self.DEFAULTS:
                continue
            if k in {"ENABLE_CRITIC", "PROMPT_USER", "DEBUG", "FILES_OFF"}:
                self.values[k] = _parse_bool(v, self.DEFAULTS[k])
            elif k in {
                "MAX_CONTEXT_SIZE", "MAX_MEMORY_ITEM_SIZE",
                "SUMMARIZER_CHUNK_SIZE", "MAX_STEPS", "REASONING_EFFORT",
                "BITNET_THREADS", "BITNET_N_PREDICT",
            }:
                self.values[k] = _parse_int(v, self.DEFAULTS[k])
            elif k in {"BITNET_BITS", "BITNET_TEMP"}:
                try:
                    self.values[k] = float(v)
                except (TypeError, ValueError):
                    self.values[k] = self.DEFAULTS[k]
            else:
                self.values[k] = "" if v is None else str(v)
        model = str(self.values.get("MODEL") or "")
        summ = str(self.values.get("SUMMARIZER_MODEL") or "")
        cloudish = ("gpt-", "openai", "chatgpt", "o1", "o3", "claude")
        if not model or any(x in model.lower() for x in cloudish):
            self.values["MODEL"] = BITNET_MODEL
        if not summ or any(x in summ.lower() for x in cloudish):
            self.values["SUMMARIZER_MODEL"] = BITNET_SUMMARIZER
        backend = str(self.values.get("BITNET_BACKEND") or "auto").strip().lower()
        if backend not in {"auto", "cpp", "hf", "stub"}:
            self.values["BITNET_BACKEND"] = "auto"

    def apply_work_dir(self):
        work = (self.values.get("WORK_DIR") or "").strip()
        if not work:
            work = str(Path.home() / "catr1-miniagi")
            self.values["WORK_DIR"] = work
        try:
            Path(work).mkdir(parents=True, exist_ok=True)
        except OSError:
            # Fall back to project-local work dir if home is not writable.
            fallback = str(_ROOT / "work")
            self.values["WORK_DIR"] = fallback
            Path(fallback).mkdir(parents=True, exist_ok=True)

    def save_env(self, path: Path = ENV_FILE):
        lines = [
            "# CatSeek R1 — real BitNet b1.58 (no OpenAI)",
            f'MODEL="{self.values.get("MODEL", BITNET_MODEL)}"',
            f'SUMMARIZER_MODEL="{self.values.get("SUMMARIZER_MODEL", BITNET_SUMMARIZER)}"',
            f'BITNET_BITS={self.values.get("BITNET_BITS", BITNET_BITS)}',
            f'BITNET_HOME="{self.values.get("BITNET_HOME", DEFAULT_BITNET_HOME)}"',
            f'BITNET_GGUF="{self.values.get("BITNET_GGUF", DEFAULT_BITNET_GGUF)}"',
            f'BITNET_BACKEND={self.values.get("BITNET_BACKEND", "auto")}',
            f'BITNET_THREADS={self.values.get("BITNET_THREADS", 2)}',
            f'BITNET_TEMP={self.values.get("BITNET_TEMP", 0.7)}',
            f'BITNET_N_PREDICT={self.values.get("BITNET_N_PREDICT", 256)}',
            f'ENABLE_CRITIC={"true" if self.values["ENABLE_CRITIC"] else "false"}',
            f'PROMPT_USER={"true" if self.values["PROMPT_USER"] else "false"}',
            "",
            f'MAX_CONTEXT_SIZE={self.values["MAX_CONTEXT_SIZE"]}',
            f'MAX_MEMORY_ITEM_SIZE={self.values["MAX_MEMORY_ITEM_SIZE"]}',
            f'SUMMARIZER_CHUNK_SIZE={self.values["SUMMARIZER_CHUNK_SIZE"]}',
            "",
            f'WORK_DIR={self.values.get("WORK_DIR", "")}',
            f'DEBUG={"true" if self.values["DEBUG"] else "false"}',
            "",
            f'FILES_OFF={"true" if self.values["FILES_OFF"] else "false"}',
            f'MAX_STEPS={self.values["MAX_STEPS"]}',
            f'REASONING_EFFORT={self.values["REASONING_EFFORT"]}',
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    @property
    def agent_model(self) -> str:
        return str(self.values.get("MODEL") or BITNET_MODEL)

    @property
    def summarizer_model(self) -> str:
        return str(self.values.get("SUMMARIZER_MODEL") or BITNET_SUMMARIZER)

    @property
    def bitnet_bits(self) -> float:
        try:
            return float(self.values.get("BITNET_BITS", BITNET_BITS))
        except (TypeError, ValueError):
            return BITNET_BITS

    @property
    def bitnet_home(self) -> Path:
        return Path(str(self.values.get("BITNET_HOME") or DEFAULT_BITNET_HOME)).expanduser()

    @property
    def bitnet_gguf(self) -> Path:
        raw = Path(str(self.values.get("BITNET_GGUF") or DEFAULT_BITNET_GGUF)).expanduser()
        if raw.is_absolute():
            return raw
        return self.bitnet_home / raw

    @property
    def bitnet_backend(self) -> str:
        return str(self.values.get("BITNET_BACKEND") or "auto").strip().lower()

    @property
    def bitnet_threads(self) -> int:
        return max(1, int(self.values.get("BITNET_THREADS") or 2))

    @property
    def bitnet_temp(self) -> float:
        try:
            return float(self.values.get("BITNET_TEMP", 0.7))
        except (TypeError, ValueError):
            return 0.7

    @property
    def bitnet_n_predict(self) -> int:
        return max(16, int(self.values.get("BITNET_N_PREDICT") or 256))

    @property
    def enable_critic(self) -> bool:
        return bool(self.values["ENABLE_CRITIC"])

    @property
    def prompt_user(self) -> bool:
        return bool(self.values["PROMPT_USER"])

    @property
    def max_context_size(self) -> int:
        return int(self.values["MAX_CONTEXT_SIZE"])

    @property
    def max_memory_item_size(self) -> int:
        return int(self.values["MAX_MEMORY_ITEM_SIZE"])

    @property
    def summarizer_chunk_size(self) -> int:
        return int(self.values["SUMMARIZER_CHUNK_SIZE"])

    @property
    def work_dir(self) -> str:
        return str(self.values.get("WORK_DIR") or "")

    @property
    def debug(self) -> bool:
        return bool(self.values["DEBUG"])

    @property
    def files_off(self) -> bool:
        return bool(self.values["FILES_OFF"])

    @property
    def max_steps(self) -> int:
        return int(self.values["MAX_STEPS"])

    @property
    def reasoning_effort(self) -> int:
        return max(1, min(100, int(self.values["REASONING_EFFORT"])))

    def char_budget(self, tokens: int) -> int:
        # Offline stand-in: ~4 chars/token like MiniAGI tiktoken budgets.
        return max(256, int(tokens) * 4)


def write_env_example(path: Path = ENV_EXAMPLE):
    path.write_text(
        "\n".join(
            [
                "# CatSeek R1 — real BitNet b1.58 (no OpenAI / no cloud LLM)",
                f'MODEL="{BITNET_MODEL}"',
                f'SUMMARIZER_MODEL="{BITNET_SUMMARIZER}"',
                f"BITNET_BITS={BITNET_BITS}",
                f'BITNET_HOME="{DEFAULT_BITNET_HOME}"',
                f'BITNET_GGUF="{DEFAULT_BITNET_GGUF}"',
                "BITNET_BACKEND=auto",
                f"BITNET_THREADS={max(1, (os.cpu_count() or 2) // 2)}",
                "BITNET_TEMP=0.7",
                "BITNET_N_PREDICT=256",
                "ENABLE_CRITIC=false",
                "PROMPT_USER=true",
                "",
                "MAX_CONTEXT_SIZE=4000",
                "MAX_MEMORY_ITEM_SIZE=2000",
                "SUMMARIZER_CHUNK_SIZE=3000",
                "",
                "WORK_DIR=",
                "DEBUG=false",
                "",
                "FILES_OFF=true",
                "MAX_STEPS=12",
                "REASONING_EFFORT=64",
                "",
            ]
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Real BitNet b1.58 — ternary math + inference backends
# ---------------------------------------------------------------------------

class BitLinear:
    """BitNet b1.58 BitLinear: absmean-quantized ternary weights W ∈ {-1,0,1}.

    Matches the W1.58A8 scheme (Wang et al. / Microsoft BitNet):
      γ = mean(|W|);  W̃ = RoundClip(W/γ, -1, 1);  y = (X @ W̃ᵀ) * γ
    """

    @staticmethod
    def round_clip(x: list[list[float]], lo: float = -1.0, hi: float = 1.0) -> list[list[float]]:
        out = []
        for row in x:
            out.append([max(lo, min(hi, float(round(v)))) for v in row])
        return out

    @staticmethod
    def absmean(weight: list[list[float]]) -> float:
        flat = [abs(v) for row in weight for v in row]
        if not flat:
            return 1.0
        return max(1e-8, sum(flat) / len(flat))

    @classmethod
    def quantize(cls, weight: list[list[float]]) -> tuple[list[list[float]], float]:
        gamma = cls.absmean(weight)
        scaled = [[v / gamma for v in row] for row in weight]
        return cls.round_clip(scaled), gamma

    @classmethod
    def forward(cls, x: list[list[float]], weight: list[list[float]]) -> list[list[float]]:
        w_q, gamma = cls.quantize(weight)
        # y = x @ w_q.T * gamma
        out_dim = len(w_q)
        in_dim = len(w_q[0]) if w_q else 0
        result = []
        for row in x:
            if len(row) != in_dim:
                raise ValueError(f"BitLinear: expected in_dim={in_dim}, got {len(row)}")
            y = []
            for j in range(out_dim):
                s = 0.0
                wj = w_q[j]
                for k, xv in enumerate(row):
                    s += xv * wj[k]
                y.append(s * gamma)
            result.append(y)
        return result

    @classmethod
    def bits_per_weight(cls) -> float:
        # ternary alphabet size 3 → log2(3) ≈ 1.58496 bits
        return math.log2(3.0)


def _find_llama_cli(bitnet_home: Path) -> Path | None:
    candidates = [
        bitnet_home / "build" / "bin" / "llama-cli",
        bitnet_home / "build" / "bin" / "Release" / "llama-cli.exe",
        bitnet_home / "build" / "bin" / "llama-cli.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    which = shutil.which("llama-cli")
    return Path(which) if which else None


def _strip_llama_noise(text: str) -> str:
    lines = []
    skip_prefixes = (
        "llama_", "ggml_", "main:", "system_info:", "sampling:",
        "generate:", "slot ", "print_info:", "load_", "llm_",
        "common_", "gguf_", "ctrl+c",
    )
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if not s:
            continue
        if any(low.startswith(p) for p in skip_prefixes):
            continue
        if re.match(r"^\[\d+/\d+\]", s):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class BitNetEngine:
    """Microsoft BitNet b1.58 inference: bitnet.cpp preferred, HF transformers fallback."""

    def __init__(self, params: "MiniAGIParams"):
        self.params = params
        self.debug = params.debug
        self._backend: str | None = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._resolve_backend()

    def _resolve_backend(self):
        pref = self.params.bitnet_backend
        if pref == "stub":
            self._backend = "stub"
            return
        if pref in {"auto", "cpp"}:
            cli = _find_llama_cli(self.params.bitnet_home)
            gguf = self.params.bitnet_gguf
            if cli and gguf.exists():
                self._backend = "cpp"
                return
            if pref == "cpp":
                self._backend = "stub"
                return
        if pref in {"auto", "hf"}:
            if self._try_load_hf():
                self._backend = "hf"
                return
            if pref == "hf":
                self._backend = "stub"
                return
        self._backend = "stub"

    def _try_load_hf(self) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return False
        model_id = self.params.agent_model or BITNET_MODEL
        try:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
            )
            self._hf_model.eval()
            return True
        except Exception as exc:  # noqa: BLE001
            if self.debug:
                print(f"[DEBUG] HF BitNet load failed: {exc}")
            self._hf_model = None
            self._hf_tokenizer = None
            return False

    @property
    def backend(self) -> str:
        return self._backend or "stub"

    @property
    def is_live(self) -> bool:
        return self.backend in {"cpp", "hf"}

    def status(self) -> str:
        cli = _find_llama_cli(self.params.bitnet_home)
        gguf = self.params.bitnet_gguf
        return (
            f"backend={self.backend} bits={BitLinear.bits_per_weight():.2f} "
            f"home={self.params.bitnet_home} "
            f"llama-cli={'yes' if cli else 'no'} "
            f"gguf={'yes' if gguf.exists() else 'missing:' + str(gguf)} "
            f"model={self.params.agent_model}"
        )

    def generate(
        self,
        prompt: str,
        n_predict: int | None = None,
        system: str | None = None,
        conversation: bool = False,
    ) -> str:
        """Generate text the same way microsoft/BitNet run_inference.py does.

        Flags mirrored: -m MODEL -n N -p PROMPT -t THREADS -c CTX -temp TEMP [-cnv]
        """
        n = n_predict if n_predict is not None else self.params.bitnet_n_predict
        use_cnv = conversation or bool(system)
        if self.backend == "cpp":
            # BitNet -cnv: -p is the system prompt; user turn follows interactively.
            # For one-shot MiniAGI we pack system+user into -p when not in cnv,
            # or pass system as -p with -cnv and append the user request.
            if use_cnv and system:
                return self._generate_cpp(system, n, conversation=True, user_prompt=prompt)
            full = prompt if not system else f"{system}\n\n{prompt}"
            return self._generate_cpp(full, n, conversation=False)
        if self.backend == "hf":
            full = prompt
            if system:
                full = f"System: {system}\nUser: {prompt}\nAssistant:"
            return self._generate_hf(full, n)
        raise RuntimeError(
            "No live BitNet backend. Run: python3 \"#catseeekr110.1.26.py\" --setup-bitnet"
        )

    def _generate_cpp(
        self,
        prompt: str,
        n_predict: int,
        conversation: bool = False,
        user_prompt: str | None = None,
    ) -> str:
        """Invoke BitNet exactly like run_inference.py → build/bin/llama-cli."""
        home = self.params.bitnet_home
        gguf = self.params.bitnet_gguf
        run_py = home / "run_inference.py"
        cli = _find_llama_cli(home)
        if not gguf.exists():
            raise RuntimeError(f"BitNet GGUF missing: {gguf}")

        ctx = max(512, min(self.params.max_context_size, 4096))
        # Prefer official wrapper when present (exact BitNet entrypoint).
        if run_py.exists():
            cmd = [
                sys.executable, str(run_py),
                "-m", str(gguf),
                "-n", str(n_predict),
                "-t", str(self.params.bitnet_threads),
                "-p", prompt,
                "-c", str(ctx),
                "-temp", str(self.params.bitnet_temp),
            ]
            if conversation:
                cmd.append("-cnv")
        else:
            if not cli:
                raise RuntimeError("bitnet.cpp llama-cli missing — run --setup-bitnet")
            # Same argv microsoft/BitNet/run_inference.py builds:
            cmd = [
                str(cli),
                "-m", str(gguf),
                "-n", str(n_predict),
                "-t", str(self.params.bitnet_threads),
                "-p", prompt,
                "-ngl", "0",
                "-c", str(ctx),
                "--temp", str(self.params.bitnet_temp),
                "-b", "1",
            ]
            if conversation:
                cmd.append("-cnv")

        if self.debug:
            print(f"[DEBUG] bitnet: {' '.join(cmd[:10])}…")

        # -cnv is interactive; feed user prompt on stdin then exit.
        stdin_data = None
        if conversation and user_prompt:
            stdin_data = user_prompt.strip() + "\n/exit\n"

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(home),
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=max(90, n_predict * 4),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"bitnet.cpp timed out: {exc}") from exc
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        cleaned = _strip_llama_noise(out)
        if proc.returncode != 0 and not cleaned:
            raise RuntimeError(f"bitnet.cpp exit {proc.returncode}: {_clip(out, 800)}")
        if prompt.strip() and cleaned.startswith(prompt.strip()[:80]):
            cleaned = cleaned[len(prompt.strip()):].lstrip()
        return cleaned.strip() or out.strip()

    def _generate_hf(self, prompt: str, n_predict: int) -> str:
        import torch

        assert self._hf_model is not None and self._hf_tokenizer is not None
        tok = self._hf_tokenizer
        inputs = tok(prompt, return_tensors="pt")
        try:
            device = next(self._hf_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except StopIteration:
            pass
        with torch.no_grad():
            out = self._hf_model.generate(
                **inputs,
                max_new_tokens=n_predict,
                do_sample=self.params.bitnet_temp > 0.05,
                temperature=max(0.01, self.params.bitnet_temp),
                pad_token_id=getattr(tok, "eos_token_id", None),
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()


def setup_bitnet(params: "MiniAGIParams") -> str:
    """Clone microsoft/BitNet, download GGUF, and run setup_env.py (i2_s)."""
    home = params.bitnet_home
    logs: list[str] = []

    def run(cmd: list[str], cwd: Path | None = None):
        logs.append("$ " + " ".join(cmd))
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=False
        )
        if proc.stdout:
            logs.append(proc.stdout.strip())
        if proc.stderr:
            logs.append(proc.stderr.strip())
        if proc.returncode != 0:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")

    if not home.exists():
        run(["git", "clone", "--recursive", BITNET_GIT, str(home)])
    else:
        logs.append(f"BitNet home exists: {home}")

    # Python deps for setup_env / HF download
    run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    model_dir = home / "models" / "BitNet-b1.58-2B-4T"
    model_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "-c",
        (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download('{BITNET_GGUF_REPO}', local_dir=r'{model_dir}')"
        ),
    ])

    req = home / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])

    setup_py = home / "setup_env.py"
    if not setup_py.exists():
        raise RuntimeError(f"setup_env.py missing under {home}")
    run([sys.executable, str(setup_py), "-md", str(model_dir), "-q", "i2_s"], cwd=home)

    params.values["BITNET_HOME"] = str(home)
    params.values["BITNET_GGUF"] = DEFAULT_BITNET_GGUF
    params.values["BITNET_BACKEND"] = "cpp"
    params.values["MODEL"] = BITNET_MODEL
    params.values["SUMMARIZER_MODEL"] = BITNET_SUMMARIZER
    params.save_env()
    engine = BitNetEngine(params)
    logs.append("setup complete · " + engine.status())
    return "\n".join(logs)


class BitNetCoder:
    """Code synthesis via live BitNet; domain stubs when BitNet weights are offline."""

    LANG_ALIASES = {
        "c": "c", "c++": "cpp", "cpp": "cpp", "python": "python", "py": "python",
        "javascript": "javascript", "js": "javascript", "go": "go", "rust": "rust",
        "java": "java", "bash": "bash", "shell": "bash",
        "html": "html", "htm": "html",
    }

    def __init__(self, engine: BitNetEngine | None = None, work_dir: str | Path | None = None):
        self.engine = engine
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()

    @classmethod
    def is_coding_request(cls, text: str) -> bool:
        q = text.lower()
        if any(w in q for w in ("write", "implement", "code", "program", "print", "hello", "game", "chess")):
            return True
        return any(re.search(rf"\bin\s+{re.escape(a)}\b", q) for a in cls.LANG_ALIASES)

    @classmethod
    def detect_lang(cls, text: str) -> str:
        q = text.lower()
        for alias in sorted(cls.LANG_ALIASES, key=len, reverse=True):
            if re.search(rf"\bin\s+{re.escape(alias)}\b", q):
                return cls.LANG_ALIASES[alias]
        if "python" in q or "py" in q:
            return "python"
        return "python"

    @classmethod
    def _extract_message(cls, text: str) -> str:
        m = re.search(r'["“](.+?)["”]', text) or re.search(r"'(.+?)'", text)
        if m:
            return m.group(1).strip()
        low = text.lower().strip()
        low = re.sub(r"\bin\s+[\w+#++]+\s*$", "", low).strip()
        m = re.search(r"\b(hello(?:\s+[\w'-]+){0,4})\b", low)
        if m:
            return m.group(1).strip()
        cleaned = re.sub(
            r"\b(write|implement|code|program|create|make|print|a|an|the)\b",
            " ",
            low,
        )
        return re.sub(r"\s+", " ", cleaned).strip(" .") or "hello"

    @staticmethod
    def extract_fenced_code(text: str) -> tuple[str, str]:
        """Return (lang, code) from first markdown fence, else ('', full text)."""
        m = re.search(r"```(\w+)?\n([\s\S]*?)```", text)
        if not m:
            return "", (text or "").strip()
        return (m.group(1) or "text").strip(), m.group(2).strip("\n") + "\n"

    @classmethod
    def default_filename(cls, objective: str, lang: str) -> str:
        q = objective.lower()
        if "chess" in q:
            if lang == "c":
                return "chess.c"
            if lang == "html":
                return "chess.html"
            return "chess.py" if lang in {"", "python", "py"} else f"chess.{lang}"
        if "hello" in q and lang == "c":
            return "hello_cat.c"
        if lang == "html":
            return "hello_cat.html" if "hello" in q else "index.html"
        if lang in {"c", "cpp"}:
            return "main.c" if lang == "c" else "main.cpp"
        if lang == "go":
            return "main.go"
        if lang == "rust":
            return "main.rs"
        if lang in {"javascript", "js"}:
            return "main.js"
        return "main.py"

    def synthesize(self, text: str) -> str:
        if self.engine and self.engine.is_live:
            lang = self.detect_lang(text)
            prompt = (
                f"Write a complete, runnable {lang} program for this request. "
                f"Reply with a short observation line, then a fenced ```{lang} code block only.\n"
                f"Request: {text}"
            )
            try:
                out = self.engine.generate(
                    prompt,
                    n_predict=max(512, self.engine.params.bitnet_n_predict),
                    system="You are CatSeek R1 BitNet coder. Output complete runnable code.",
                )
                if "```" in out:
                    return f"Observation: BitNet ({self.engine.backend}) code.\n\n{out}"
                return (
                    f"Observation: BitNet ({self.engine.backend}) code for `{text}`.\n\n"
                    f"```{lang}\n{out}\n```"
                )
            except Exception as exc:  # noqa: BLE001
                return self._stub(text) + f"\n\n[BitNet error → stub: {exc}]"
        return self._stub(text)

    def deliver(self, objective: str, observation: str) -> tuple[str, Path | None]:
        """Write code to WORK_DIR (file:// on device) and print it to the terminal."""
        lang, code = self.extract_fenced_code(observation)
        if not code.strip():
            print("\n===== CatSeek R1 · no code block to print =====\n", flush=True)
            print(observation, flush=True)
            return observation, None
        if not lang or lang == "text":
            lang = self.detect_lang(objective)
        name = self.default_filename(objective, lang)
        candidates = [self.work_dir, _ROOT / "work", Path.cwd()]
        path = None
        last_err = None
        for base in candidates:
            try:
                base.mkdir(parents=True, exist_ok=True)
                candidate = base / name
                candidate.write_text(code, encoding="utf-8")
                path = candidate
                break
            except OSError as exc:
                last_err = exc
                continue
        if path is None:
            file_url = ""
            saved = f"(could not write file: {last_err})"
        else:
            file_url = path.resolve().as_uri()
            saved = str(path.resolve())
        banner = (
            f"\n===== CatSeek R1 · CODE ({lang}) =====\n"
            f"# saved: {saved}\n"
            f"# open:  {file_url or '(n/a)'}\n"
            f"{code}"
            f"===== end CODE =====\n"
        )
        print(banner, flush=True)
        note = (
            f"\n\n[printed to terminal]\n"
            f"[saved on device] {saved}\n"
            f"[file url] {file_url or '(n/a)'}"
        )
        if path and lang == "python":
            note += f"\n[run] python3 \"{path}\""
        elif path and lang == "c":
            note += f"\n[run] cc \"{path}\" -o /tmp/a.out && /tmp/a.out"
        elif path and lang == "html":
            note += f"\n[open] open \"{path}\""
        return observation.rstrip() + note, path

    @classmethod
    def _stub(cls, text: str) -> str:
        lang = cls.detect_lang(text)
        q = text.lower()
        if "chess" in q and lang == "python":
            code = cls._chess_py()
            return (
                "Observation: terminal chess game in Python 3 "
                "(stub while BitNet weights offline).\n\n"
                f"```python\n{code}```"
            )
        if "chess" in q and lang == "c":
            code = cls._chess_c()
            return (
                "Observation: terminal chess starter in C "
                "(stub while BitNet weights offline).\n\n"
                f"```c\n{code}```"
            )
        msg = cls._extract_message(text)
        if lang == "html":
            safe = (
                msg.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            code = (
                "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
                "  <meta charset=\"utf-8\">\n"
                f"  <title>{safe}</title>\n"
                "  <style>body{font-family:system-ui;display:grid;place-items:center;"
                "min-height:100vh;margin:0;background:#111;color:#f5f5f5}"
                "h1{font-size:clamp(2rem,8vw,5rem)}</style>\n"
                "</head>\n<body>\n"
                f"  <h1>{safe}</h1>\n"
                "</body>\n</html>\n"
            )
            return (
                f"Observation: HTML page for `{msg}`.\n\n"
                f"```html\n{code}```"
            )
        msg_c = msg.replace('"', '\\"')
        if lang == "c":
            code = (
                f'#include <stdio.h>\n\nint main(void) {{\n'
                f'    printf("{msg_c}\\n");\n    return 0;\n}}\n'
            )
        elif lang == "go":
            code = (
                f'package main\n\nimport "fmt"\n\n'
                f'func main() {{\n    fmt.Println("{msg_c}")\n}}\n'
            )
        elif lang == "rust":
            code = f'fn main() {{\n    println!("{msg_c}");\n}}\n'
        else:
            lang = "python"
            code = f"print('{msg.replace(chr(39), chr(92)+chr(39))}')\n"
        return (
            f"Observation: STUB (no live BitNet) {lang} program for `{msg}`.\n"
            f"Run --setup-bitnet for real microsoft/bitnet-b1.58-2B-4T.\n\n"
            f"```{lang}\n{code}```"
        )

    @staticmethod
    def _chess_c() -> str:
        return r'''#include <stdio.h>

/* Minimal C chess board printer — expand later for full rules. */
static const char *START[8] = {
    "rnbqkbnr",
    "pppppppp",
    "........",
    "........",
    "........",
    "........",
    "PPPPPPPP",
    "RNBQKBNR",
};

static void print_board(void) {
    int r, c;
    puts("  a b c d e f g h");
    for (r = 0; r < 8; r++) {
        printf("%d ", 8 - r);
        for (c = 0; c < 8; c++) {
            char ch = START[r][c];
            printf("%c ", ch == '.' ? '.' : ch);
        }
        printf("%d\n", 8 - r);
    }
    puts("  a b c d e f g h");
}

int main(void) {
    puts("CatSeek chess (C) — starting position");
    print_board();
    puts("hello cat — compile with: cc chess.c -o chess && ./chess");
    return 0;
}
'''

    @staticmethod
    def _chess_py() -> str:
        # Compact playable terminal chess (Python 3) — printed to stdout via deliver().
        return r'''#!/usr/bin/env python3
"""Minimal terminal chess — Python 3. Moves like e2e4, O-O, O-O-O, resign."""

from __future__ import annotations

import re

FILES = "abcdefgh"
UNICODE = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
    ".": "·",
}


def new_board():
    return [
        list("rnbqkbnr"),
        list("pppppppp"),
        list("........"),
        list("........"),
        list("........"),
        list("........"),
        list("PPPPPPPP"),
        list("RNBQKBNR"),
    ]


def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8


def parse_sq(s):
    s = s.strip().lower()
    if len(s) != 2 or s[0] not in FILES or s[1] not in "12345678":
        raise ValueError(f"bad square: {s}")
    return 8 - int(s[1]), FILES.index(s[0])


def sq_name(r, c):
    return f"{FILES[c]}{8 - r}"


def piece_color(p):
    if p == ".":
        return None
    return "w" if p.isupper() else "b"


def find_king(board, color):
    target = "K" if color == "w" else "k"
    for r in range(8):
        for c in range(8):
            if board[r][c] == target:
                return r, c
    return None


def ray_hits(board, r, c, dr, dc, enemy):
    r += dr
    c += dc
    while in_bounds(r, c):
        p = board[r][c]
        if p == ".":
            r += dr
            c += dc
            continue
        return piece_color(p) == enemy and p.lower() in "qrb" and (
            (p.lower() == "q")
            or (p.lower() == "r" and (dr == 0 or dc == 0))
            or (p.lower() == "b" and dr != 0 and dc != 0)
        )
    return False


def square_attacked(board, r, c, by_color):
    enemy = by_color
    # pawns
    pr = 1 if enemy == "w" else -1
    for dc in (-1, 1):
        rr, cc = r + pr, c + dc
        if in_bounds(rr, cc) and board[rr][cc] == ("P" if enemy == "w" else "p"):
            return True
    # knights
    for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
        rr, cc = r + dr, c + dc
        if in_bounds(rr, cc) and board[rr][cc] == ("N" if enemy == "w" else "n"):
            return True
    # king
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if in_bounds(rr, cc) and board[rr][cc] == ("K" if enemy == "w" else "k"):
                return True
    # sliding
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        if ray_hits(board, r, c, dr, dc, enemy):
            return True
    return False


def in_check(board, color):
    k = find_king(board, color)
    if not k:
        return True
    return square_attacked(board, k[0], k[1], "b" if color == "w" else "w")


def clone(board):
    return [row[:] for row in board]


def apply_move(board, fr, fc, tr, tc, promo="Q"):
    b = clone(board)
    piece = b[fr][fc]
    b[tr][tc] = piece
    b[fr][fc] = "."
    if piece in "Pp" and tr in (0, 7):
        b[tr][tc] = promo if piece == "P" else promo.lower()
    # castling rook move
    if piece in "Kk" and abs(tc - fc) == 2:
        if tc == 6:  # king side
            b[tr][5] = b[tr][7]
            b[tr][7] = "."
        elif tc == 2:  # queen side
            b[tr][3] = b[tr][0]
            b[tr][0] = "."
    return b


def pawn_moves(board, r, c, color):
    moves = []
    direction = -1 if color == "w" else 1
    start = 6 if color == "w" else 1
    rr = r + direction
    if in_bounds(rr, c) and board[rr][c] == ".":
        moves.append((rr, c))
        rr2 = r + 2 * direction
        if r == start and board[rr2][c] == ".":
            moves.append((rr2, c))
    for dc in (-1, 1):
        cc = c + dc
        if in_bounds(rr, cc) and piece_color(board[rr][cc]) not in (None, color):
            moves.append((rr, cc))
    return moves


def sliding_moves(board, r, c, color, dirs):
    moves = []
    for dr, dc in dirs:
        rr, cc = r + dr, c + dc
        while in_bounds(rr, cc):
            col = piece_color(board[rr][cc])
            if col is None:
                moves.append((rr, cc))
            else:
                if col != color:
                    moves.append((rr, cc))
                break
            rr += dr
            cc += dc
    return moves


def piece_moves(board, r, c):
    p = board[r][c]
    color = piece_color(p)
    if not color:
        return []
    pl = p.lower()
    if pl == "p":
        return pawn_moves(board, r, c, color)
    if pl == "n":
        out = []
        for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
            rr, cc = r + dr, c + dc
            if in_bounds(rr, cc) and piece_color(board[rr][cc]) != color:
                out.append((rr, cc))
        return out
    if pl == "b":
        return sliding_moves(board, r, c, color, ((1, 1), (1, -1), (-1, 1), (-1, -1)))
    if pl == "r":
        return sliding_moves(board, r, c, color, ((1, 0), (-1, 0), (0, 1), (0, -1)))
    if pl == "q":
        return sliding_moves(
            board, r, c, color,
            ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)),
        )
    if pl == "k":
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if in_bounds(rr, cc) and piece_color(board[rr][cc]) != color:
                    out.append((rr, cc))
        # castling (no check history — simplified: empty path + not in check)
        if not in_check(board, color):
            row = 7 if color == "w" else 0
            rook = "R" if color == "w" else "r"
            if r == row and c == 4:
                if board[row][5] == board[row][6] == "." and board[row][7] == rook:
                    if not square_attacked(board, row, 5, "b" if color == "w" else "w") and not square_attacked(
                        board, row, 6, "b" if color == "w" else "w"
                    ):
                        out.append((row, 6))
                if board[row][1] == board[row][2] == board[row][3] == "." and board[row][0] == rook:
                    if not square_attacked(board, row, 3, "b" if color == "w" else "w") and not square_attacked(
                        board, row, 2, "b" if color == "w" else "w"
                    ):
                        out.append((row, 2))
        return out
    return []


def legal_moves(board, color):
    moves = []
    for r in range(8):
        for c in range(8):
            if piece_color(board[r][c]) != color:
                continue
            for tr, tc in piece_moves(board, r, c):
                nxt = apply_move(board, r, c, tr, tc)
                if not in_check(nxt, color):
                    moves.append((r, c, tr, tc))
    return moves


def render(board):
    print("\n    a b c d e f g h")
    print("  +-----------------+")
    for r in range(8):
        cells = " ".join(UNICODE.get(board[r][c], board[r][c]) for c in range(8))
        print(f"{8 - r} | {cells} | {8 - r}")
    print("  +-----------------+")
    print("    a b c d e f g h\n")


def parse_move(text, board, color):
    text = text.strip()
    if text.lower() in {"o-o", "0-0"}:
        row = 7 if color == "w" else 0
        return row, 4, row, 6
    if text.lower() in {"o-o-o", "0-0-0"}:
        row = 7 if color == "w" else 0
        return row, 4, row, 2
    m = re.match(r"^([a-h][1-8])([a-h][1-8])([qrbn])?$", text.lower())
    if not m:
        raise ValueError("use e2e4 / O-O / O-O-O / resign")
    fr, fc = parse_sq(m.group(1))
    tr, tc = parse_sq(m.group(2))
    return fr, fc, tr, tc


def main():
    board = new_board()
    turn = "w"
    print("CatSeek terminal chess. Moves: e2e4, O-O, O-O-O, resign, help")
    while True:
        render(board)
        side = "White" if turn == "w" else "Black"
        if in_check(board, turn):
            print(f"{side} is in check.")
        legal = legal_moves(board, turn)
        if not legal:
            print("Checkmate!" if in_check(board, turn) else "Stalemate!")
            break
        raw = input(f"{side}> ").strip()
        if not raw:
            continue
        low = raw.lower()
        if low in {"resign", "quit", "exit"}:
            print(f"{side} resigns.")
            break
        if low == "help":
            print("Enter from-to squares like e2e4. Castling: O-O / O-O-O.")
            continue
        try:
            fr, fc, tr, tc = parse_move(raw, board, turn)
        except ValueError as exc:
            print(exc)
            continue
        if (fr, fc, tr, tc) not in legal:
            print("Illegal move.")
            continue
        board = apply_move(board, fr, fc, tr, tc)
        turn = "b" if turn == "w" else "w"


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Persistent memory
# ---------------------------------------------------------------------------

class PersistentMemory:
    def __init__(self, path: Path = MEMORY_FILE):
        self.path = Path(path)
        self.first_boot = False
        self.data = {
            "version": MEMORY_VERSION,
            "model": AGENT_NAME,
            "first_seen": None,
            "updated": None,
            "objective": "write hello cat in c",
            "log": [],
            "summarized_history": "",
            "action_memory": [],
            "miniagi_params": {},
        }
        self.load_or_create()

    def load_or_create(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.first_boot = True
            self.data["first_seen"] = _utc_now()
            self.save()
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.data.update(raw)
        except (OSError, json.JSONDecodeError, TypeError):
            self.first_boot = True
            self.data["first_seen"] = _utc_now()
        self.save()

    def save(self):
        self.data["updated"] = _utc_now()
        self.data["log"] = list(self.data.get("log") or [])[-MAX_LOG_ENTRIES:]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def append_log(self, who: str, text: str):
        self.data.setdefault("log", []).append({"who": who, "text": text.strip()})
        self.save()

    def clear(self):
        first = self.data.get("first_seen") or _utc_now()
        kept_params = self.data.get("miniagi_params") or {}
        self.data = {
            "version": MEMORY_VERSION,
            "model": AGENT_NAME,
            "first_seen": first,
            "updated": _utc_now(),
            "objective": "write hello cat in c",
            "log": [],
            "summarized_history": "",
            "action_memory": [],
            "miniagi_params": kept_params,
            "cleared_at": _utc_now(),
        }
        self.save()


# ---------------------------------------------------------------------------
# MiniAGI-compatible agent
# ---------------------------------------------------------------------------

class CatSeekR1MiniAGI:
    """MiniAGI loop powered by real BitNet b1.58 (microsoft/BitNet)."""

    PLANNER_SYSTEM = (
        "You are CatSeek R1, a MiniAGI agent running on BitNet b1.58. "
        "Reply with ONLY one JSON object, no markdown fences:\n"
        '{"thought":"brief plan","command":"<cmd>","arg":"<argument>"}\n'
        f"Allowed commands: {', '.join(ALL_COMMANDS)}. "
        "Prefer write_code for coding tasks, answer for facts, done when finished, "
        "memorize_thoughts to record a plan, talk_to_user to ask the user."
    )

    def __init__(self, objective: str, params: MiniAGIParams):
        self.params = params
        self.objective = objective.strip()
        self.agent_model = params.agent_model
        self.summarizer_model = params.summarizer_model
        self.max_context_size = params.max_context_size
        self.max_memory_item_size = params.max_memory_item_size
        self.summarizer_chunk_size = params.summarizer_chunk_size
        self.enable_critic = params.enable_critic
        self.prompt_user = params.prompt_user
        self.work_dir = params.work_dir
        self.debug = params.debug
        self.files_off = params.files_off
        self.max_steps = params.max_steps
        self.reasoning_effort = params.reasoning_effort

        self.engine = BitNetEngine(params)
        self.coder = BitNetCoder(self.engine, work_dir=self.work_dir)
        self.summarized_history = ""
        self.action_memory: list[str] = []
        self.criticism = ""
        self.thought = ""
        self.proposed_command = ""
        self.proposed_arg = ""
        self.step = 0
        self.last_observation = ""
        self.pending_user_gate = False  # PROMPT_USER pause
        self.code_delivered = False  # session-local; ignore stale memory fences
        self.last_code_path: str | None = None

        if self.debug:
            print(
                f"[DEBUG] model={self.agent_model} summarizer={self.summarizer_model} "
                f"ctx={self.max_context_size} mem_item={self.max_memory_item_size} "
                f"chunk={self.summarizer_chunk_size} critic={self.enable_critic} "
                f"prompt_user={self.prompt_user} work_dir={self.work_dir} "
                f"files_off={self.files_off} bitnet={self.engine.status()}"
            )

    def restore_memory(self, summarized: str, actions: list[str]):
        self.summarized_history = summarized or ""
        self.action_memory = list(actions or [])

    def export_memory(self):
        return self.summarized_history, list(self.action_memory)

    def _ctx_char_limit(self) -> int:
        return self.params.char_budget(self.max_context_size)

    def _item_char_limit(self) -> int:
        return self.params.char_budget(self.max_memory_item_size)

    def _chunk_char_limit(self) -> int:
        return self.params.char_budget(self.summarizer_chunk_size)

    def _context(self) -> str:
        recent = "\n".join(self.action_memory[-16:])
        parts = [
            f"SUMMARY\n{_clip(self.summarized_history, self._chunk_char_limit())}"
            if self.summarized_history else "SUMMARY\n(none)",
            f"PREV ACTIONS:\n{recent}" if recent else "PREV ACTIONS:\n(none)",
        ]
        if self.criticism:
            parts.append(f"CRITICISM:\n{self.criticism}")
        ctx = "\n\n".join(parts)
        if self.debug:
            print(f"[DEBUG] context_chars={len(ctx)} limit={self._ctx_char_limit()}")
        return _clip(ctx, self._ctx_char_limit())

    def _summarize(self, action: str, observation: str):
        obs = _clip(observation, self._item_char_limit())
        if len(observation) > self._item_char_limit():
            # MiniAGI: chunked summarize oversized observations via BitNet when live
            if self.engine.is_live:
                try:
                    obs = _clip(
                        self.engine.generate(
                            f"Summarize for agent memory:\n{observation}",
                            n_predict=min(128, self.params.bitnet_n_predict),
                            system=f"You are {self.summarizer_model}. Be concise.",
                        ),
                        self._chunk_char_limit(),
                    )
                except Exception:  # noqa: BLE001
                    obs = _clip(
                        f"[summarized to SUMMARIZER_CHUNK_SIZE={self.summarizer_chunk_size}]\n"
                        + observation,
                        self._chunk_char_limit(),
                    )
            else:
                obs = _clip(
                    f"[summarized to SUMMARIZER_CHUNK_SIZE={self.summarizer_chunk_size}]\n"
                    + observation,
                    self._chunk_char_limit(),
                )
        if "memorize_thoughts" in action:
            new_memory = f"ACTION:\nmemorize_thoughts\nTHOUGHTS:\n{obs}\n"
        else:
            new_memory = f"ACTION:\n{action}\nRESULT:\n{obs}\n"
        self.action_memory.append(_clip(new_memory, self._item_char_limit()))
        bullet = f"- step {self.step}: {action.splitlines()[0][:120]} [{self.agent_model}]"
        if self.summarized_history:
            self.summarized_history = _clip(
                self.summarized_history + "\n" + bullet, self._item_char_limit()
            )
        else:
            self.summarized_history = (
                f"Objective: {self.objective}\n"
                f"Models: agent={self.agent_model} summarizer={self.summarizer_model}\n"
                f"BitNet: {self.engine.backend}\n"
                f"{bullet}"
            )

    def _think_heuristic(self):
        q = self.objective.lower()
        coding = BitNetCoder.is_coding_request(self.objective)
        # Only trust code produced in THIS run — stale memory used to skip write_code.
        already_coded = self.code_delivered and "```" in (self.last_observation or "")

        if self.step == 1 and not coding:
            self.thought = (
                f"Organize objective using BitNet agent_model={self.agent_model} "
                f"({self.engine.backend}); memory budgets ctx={self.max_context_size}."
            )
            self.proposed_command = "memorize_thoughts"
            self.proposed_arg = (
                f"Objective: {self.objective}\n"
                f"WORK_DIR: {self.work_dir}\n"
                f"BitNet: {self.engine.status()}\n"
                f"ENABLE_CRITIC={self.enable_critic} PROMPT_USER={self.prompt_user} "
                f"DEBUG={self.debug} FILES_OFF={self.files_off}\n"
                f"Plan: think → act → (critic) → done."
            )
            return

        # Coding objectives: write_code immediately (skip memorize so output isn't lost).
        if coding and not already_coded:
            self.thought = "Coding objective — write_code and print result to terminal."
            self.proposed_command = "write_code"
            self.proposed_arg = self.objective
            # Do not pause for PROMPT_USER — user asked for terminal code output.
            self.pending_user_gate = False
            return

        if already_coded:
            self.thought = "Code printed — send done."
            self.proposed_command = "done"
            path = self.last_code_path or "(see terminal)"
            self.proposed_arg = (
                f"Code delivered to terminal and saved at {path}.\n"
                f"{_clip(self.last_observation, 1200)}"
            )
            return

        if any(w in q for w in ("who are you", "hello", "hi", "hey")) and len(q.split()) <= 4:
            self.thought = "Greeting / identity."
            self.proposed_command = "talk_to_user" if self.prompt_user else "answer"
            self.proposed_arg = (
                f"I'm {AGENT_NAME} on real BitNet b1.58 ({self.engine.backend}). "
                f"MODEL={self.agent_model}, {self.engine.status()}."
            )
            return

        if self.step >= self.max_steps - 1:
            self.thought = "Step budget nearly spent."
            self.proposed_command = "done"
            self.proposed_arg = self.last_observation or "Reached MAX_STEPS."
            return

        ctx = self._context().lower()
        if "memorize_thoughts" not in ctx:
            self.thought = "Internal plan before answering."
            self.proposed_command = "memorize_thoughts"
            self.proposed_arg = f"Break down: {self.objective}"
            return

        self.thought = "Direct answer toward the objective."
        self.proposed_command = "answer"
        self.proposed_arg = (
            f"Toward `{self.objective}`: smallest reversible step, verify, iterate. "
            f"(BitNet={self.engine.backend} summarizer={self.summarizer_model})"
        )
        if self.prompt_user:
            self.pending_user_gate = True

    def think(self):
        self.step += 1
        self.pending_user_gate = False

        # Coding path is deterministic — never let a live/stub planner skip write_code.
        if BitNetCoder.is_coding_request(self.objective) and not self.code_delivered:
            self._think_heuristic()
            return

        ctx = self._context()
        # Live BitNet planner (same model stack as microsoft/BitNet).
        if self.engine.is_live and self.step > 1:
            n_tok = max(64, int(self.params.bitnet_n_predict * (self.reasoning_effort / 100.0)))
            user = (
                f"Objective: {self.objective}\n"
                f"Step: {self.step}/{self.max_steps}\n"
                f"FILES_OFF={self.files_off}\n"
                f"code_delivered={self.code_delivered}\n"
                f"Last observation:\n{_clip(self.last_observation, 800)}\n\n"
                f"Context:\n{ctx}\n\n"
                "Return the next MiniAGI action as JSON."
            )
            try:
                raw = self.engine.generate(
                    user,
                    n_predict=n_tok,
                    system=self.PLANNER_SYSTEM,
                    conversation=True,
                )
                obj = _extract_json_object(raw) or {}
                cmd = str(obj.get("command") or "").strip()
                if cmd in ALL_COMMANDS:
                    # Never allow premature done before code is delivered on coding tasks.
                    if cmd == "done" and BitNetCoder.is_coding_request(self.objective) and not self.code_delivered:
                        cmd = "write_code"
                        obj["arg"] = self.objective
                        obj["thought"] = "Override: must write_code before done."
                    self.thought = str(obj.get("thought") or f"BitNet({self.engine.backend}) plan")
                    self.proposed_command = cmd
                    self.proposed_arg = str(obj.get("arg") or "")
                    if cmd == "write_code":
                        self.pending_user_gate = False
                    elif self.prompt_user and cmd in {"answer", "execute_python", "execute_shell"}:
                        self.pending_user_gate = True
                    return
            except Exception as exc:  # noqa: BLE001
                if self.debug:
                    print(f"[DEBUG] BitNet think failed: {exc}")

        self._think_heuristic()

    def criticize(self) -> str:
        if not self.enable_critic:
            self.criticism = ""
            return ""
        issues = []
        if self.proposed_command == "memorize_thoughts" and self.step > 3:
            issues.append("Too much self-talk — take a real action.")
        if self.proposed_command in {"execute_python", "execute_shell", "web_search", "ingest_data", "process_data"} and self.files_off:
            issues.append(f"`{self.proposed_command}` blocked while FILES_OFF=true.")
        if self.proposed_command not in ALL_COMMANDS:
            issues.append(f"Unknown command `{self.proposed_command}`.")
        if not self.proposed_arg.strip() and self.proposed_command != "done":
            issues.append("Empty argument.")

        if self.engine.is_live and not issues:
            try:
                self.criticism = self.engine.generate(
                    (
                        f"Critique this MiniAGI action for objective `{self.objective}`:\n"
                        f"command={self.proposed_command}\narg={_clip(self.proposed_arg, 400)}\n"
                        "Reply with one short criticism paragraph."
                    ),
                    n_predict=96,
                    system=f"You are critic model {self.summarizer_model} on BitNet b1.58.",
                )
                if self.criticism:
                    return self.criticism
            except Exception:  # noqa: BLE001
                pass

        if not issues:
            self.criticism = (
                f"Criticism ({self.summarizer_model}/{self.engine.backend}): "
                "Action looks reasonable. Proceed."
            )
        else:
            self.criticism = "Criticism: " + " ".join(issues) + "\nRecommended: course-correct."
        return self.criticism

    def read_mind(self) -> tuple[str, str, str]:
        arg = self.proposed_arg
        shown = arg if len(arg) < 96 else arg[:96] + "..."
        return self.thought, self.proposed_command, shown.replace("\n", "\\n")

    def _blocked_tool(self, name: str) -> str:
        return (
            f"Error: `{name}` is a MiniAGI tool but FILES_OFF=true in CatSeek R1. "
            "Set FILES_OFF=false in Settings / .env to enable (shell/python/web/ingest). "
            "Prefer write_code / answer / memorize_thoughts / talk_to_user / done."
        )

    def act(self) -> dict:
        command = self.proposed_command
        arg = self.proposed_arg

        if command in {"execute_python", "execute_shell", "web_search", "ingest_data", "process_data"}:
            if self.files_off:
                obs = self._blocked_tool(command)
            else:
                obs = (
                    f"Observation: `{command}` requested with FILES_OFF=false. "
                    "Live shell/python/web adapters are not bundled in this CatSeek build; "
                    "use an external MiniAGI install for those tools, or write_code locally.\n"
                    f"Arg was:\n{_clip(arg, 500)}"
                )
            self._summarize(f"{command}\n{arg}", obs)
            self.criticism = ""
            self.last_observation = obs
            return {
                "thought": self.thought, "command": command, "arg": arg,
                "observation": obs, "action": "ANSWER",
            }

        if command == "memorize_thoughts":
            obs = arg.strip() or "(empty thoughts)"
            self._summarize(f"{command}\n{arg}", obs)
            self.last_observation = obs
            self.criticism = ""
            return {
                "thought": self.thought, "command": command, "arg": arg,
                "observation": obs, "action": "THINK",
            }

        if command == "write_code":
            obs = self.coder.synthesize(arg or self.objective)
            obs, path = self.coder.deliver(self.objective, obs)
            self.code_delivered = True
            self.last_code_path = str(path) if path else None
            self._summarize(f"{command}\n{arg}", obs)
            self.last_observation = obs
            self.criticism = ""
            return {
                "thought": self.thought, "command": command, "arg": arg,
                "observation": obs, "action": "ANSWER",
            }

        if command == "answer":
            if self.engine.is_live and not arg.strip():
                try:
                    arg = self.engine.generate(
                        self.objective,
                        system="You are CatSeek R1 on BitNet b1.58. Answer directly.",
                        conversation=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    arg = f"(BitNet answer failed: {exc})"
            obs = arg.strip()
            self._summarize(f"{command}\n{arg}", obs)
            self.last_observation = obs
            self.criticism = ""
            return {
                "thought": self.thought, "command": command, "arg": arg,
                "observation": obs, "action": "ANSWER",
            }

        if command == "talk_to_user":
            obs = arg.strip()
            self._summarize(f"{command}\n{arg}", obs)
            self.last_observation = obs
            self.criticism = ""
            return {
                "thought": self.thought, "command": command, "arg": arg,
                "observation": obs, "action": "ASK",
            }

        if command == "done":
            obs = arg.strip() or self.last_observation or "Done."
            self._summarize(f"{command}\n{arg}", obs)
            self.last_observation = obs
            self.criticism = ""
            return {
                "thought": self.thought, "command": command, "arg": arg,
                "observation": obs, "action": "DONE",
            }

        obs = f"Unknown command: {command}"
        self._summarize(f"{command}\n{arg}", obs)
        self.last_observation = obs
        return {
            "thought": self.thought, "command": command, "arg": arg,
            "observation": obs, "action": "ANSWER",
        }

    def user_response(self, response: str):
        self._summarize(
            f"{self.proposed_command}\n{self.proposed_arg}",
            f"User: {response}",
        )
        self.criticism = ""
        self.pending_user_gate = False
        self.last_observation = response


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CatSeekGUI(tk.Tk):
    def __init__(self, initial_objective: str | None = None):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("960x760")
        self.minsize(760, 560)

        write_env_example()
        self.params = MiniAGIParams.from_env_file()
        self.memory = PersistentMemory()
        if self.memory.data.get("miniagi_params"):
            self.params.update(self.memory.data["miniagi_params"])
            self.params.apply_work_dir()

        self.events: queue.Queue = queue.Queue()
        self.running = False
        self.agent: CatSeekR1MiniAGI | None = None
        self._restoring = False
        self._awaiting_prompt_user = False

        self.status = tk.StringVar(value=self._ready_status())
        self._build_ui(initial_objective)
        self._restore_memory()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_events)
        if initial_objective:
            self.after(250, self.start_agent)

    def _ready_status(self) -> str:
        eng = BitNetEngine(self.params)
        return (
            f"Ready · {AGENT_NAME} · BitNet={eng.backend} · "
            f"MODEL={self.params.agent_model} · "
            f"CRITIC={'on' if self.params.enable_critic else 'off'} · "
            f"PROMPT_USER={'on' if self.params.prompt_user else 'off'} · "
            f"FILES_OFF={'on' if self.params.files_off else 'off'}"
        )

    def _build_ui(self, initial_objective: str | None):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        title = ttk.Frame(outer)
        title.pack(fill="x")
        ttk.Label(title, text="CatSeek R1 · real BitNet b1.58", font=("TkDefaultFont", 18, "bold")).pack(side="left")
        ttk.Label(title, text="microsoft/BitNet · run_inference · Kondo").pack(side="right")

        ttk.Label(
            outer,
            text="Exact BitNet stack: BITNET_HOME + GGUF → run_inference.py (-m -n -p -t -c -temp -cnv) · no OpenAI",
        ).pack(anchor="w", pady=(4, 0))

        # --- MiniAGI settings panel ---
        settings = ttk.LabelFrame(outer, text="BitNet / MiniAGI parameters", padding=8)
        settings.pack(fill="x", pady=(10, 6))

        self.var_model = tk.StringVar(value=self.params.agent_model)
        self.var_summarizer = tk.StringVar(value=self.params.summarizer_model)
        self.var_workdir = tk.StringVar(value=self.params.work_dir)
        self.var_home = tk.StringVar(value=str(self.params.bitnet_home))
        self.var_gguf = tk.StringVar(value=str(self.params.values.get("BITNET_GGUF") or DEFAULT_BITNET_GGUF))
        self.var_backend = tk.StringVar(value=self.params.bitnet_backend)
        self.var_threads = tk.IntVar(value=self.params.bitnet_threads)
        self.var_temp = tk.DoubleVar(value=self.params.bitnet_temp)
        self.var_npred = tk.IntVar(value=self.params.bitnet_n_predict)
        self.var_ctx = tk.IntVar(value=self.params.max_context_size)
        self.var_mem = tk.IntVar(value=self.params.max_memory_item_size)
        self.var_chunk = tk.IntVar(value=self.params.summarizer_chunk_size)
        self.var_steps = tk.IntVar(value=self.params.max_steps)
        self.var_effort = tk.IntVar(value=self.params.reasoning_effort)
        self.var_critic = tk.BooleanVar(value=self.params.enable_critic)
        self.var_prompt_user = tk.BooleanVar(value=self.params.prompt_user)
        self.var_debug = tk.BooleanVar(value=self.params.debug)
        self.var_files_off = tk.BooleanVar(value=self.params.files_off)

        row1 = ttk.Frame(settings)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="MODEL").pack(side="left")
        ttk.Entry(row1, textvariable=self.var_model, width=28).pack(side="left", padx=4)
        ttk.Label(row1, text="SUMMARIZER").pack(side="left", padx=(8, 0))
        ttk.Entry(row1, textvariable=self.var_summarizer, width=28).pack(side="left", padx=4)
        ttk.Label(row1, text="BACKEND").pack(side="left", padx=(8, 0))
        ttk.Entry(row1, textvariable=self.var_backend, width=8).pack(side="left", padx=4)

        row1b = ttk.Frame(settings)
        row1b.pack(fill="x", pady=2)
        ttk.Label(row1b, text="BITNET_HOME").pack(side="left")
        ttk.Entry(row1b, textvariable=self.var_home).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Label(row1b, text="GGUF").pack(side="left")
        ttk.Entry(row1b, textvariable=self.var_gguf, width=36).pack(side="left", padx=4)

        row1c = ttk.Frame(settings)
        row1c.pack(fill="x", pady=2)
        ttk.Label(row1c, text="-t THREADS").pack(side="left")
        ttk.Entry(row1c, textvariable=self.var_threads, width=6).pack(side="left", padx=4)
        ttk.Label(row1c, text="-temp").pack(side="left", padx=(8, 0))
        ttk.Entry(row1c, textvariable=self.var_temp, width=6).pack(side="left", padx=4)
        ttk.Label(row1c, text="-n N_PREDICT").pack(side="left", padx=(8, 0))
        ttk.Entry(row1c, textvariable=self.var_npred, width=6).pack(side="left", padx=4)

        row2 = ttk.Frame(settings)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="MAX_CONTEXT_SIZE").pack(side="left")
        ttk.Entry(row2, textvariable=self.var_ctx, width=8).pack(side="left", padx=4)
        ttk.Label(row2, text="MAX_MEMORY_ITEM_SIZE").pack(side="left", padx=(8, 0))
        ttk.Entry(row2, textvariable=self.var_mem, width=8).pack(side="left", padx=4)
        ttk.Label(row2, text="SUMMARIZER_CHUNK_SIZE").pack(side="left", padx=(8, 0))
        ttk.Entry(row2, textvariable=self.var_chunk, width=8).pack(side="left", padx=4)
        ttk.Label(row2, text="MAX_STEPS").pack(side="left", padx=(8, 0))
        ttk.Entry(row2, textvariable=self.var_steps, width=6).pack(side="left", padx=4)

        row3 = ttk.Frame(settings)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="WORK_DIR").pack(side="left")
        ttk.Entry(row3, textvariable=self.var_workdir).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Label(row3, text="REASONING_EFFORT").pack(side="left")
        ttk.Entry(row3, textvariable=self.var_effort, width=6).pack(side="left", padx=4)

        row4 = ttk.Frame(settings)
        row4.pack(fill="x", pady=2)
        ttk.Checkbutton(row4, text="ENABLE_CRITIC", variable=self.var_critic).pack(side="left")
        ttk.Checkbutton(row4, text="PROMPT_USER", variable=self.var_prompt_user).pack(side="left", padx=8)
        ttk.Checkbutton(row4, text="DEBUG", variable=self.var_debug).pack(side="left")
        ttk.Checkbutton(row4, text="FILES_OFF", variable=self.var_files_off).pack(side="left", padx=8)
        ttk.Button(row4, text="Save .env", command=self.save_params).pack(side="right")

        ttk.Label(outer, text="Objective").pack(anchor="w", pady=(8, 3))
        self.objective = tk.Text(outer, height=3, wrap="word")
        self.objective.pack(fill="x")
        seed = initial_objective or self.memory.data.get("objective") or "write hello cat in c"
        self.objective.insert("1.0", seed)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=8)
        self.run_btn = ttk.Button(buttons, text="Run Agent", command=self.start_agent)
        self.run_btn.pack(side="left")
        ttk.Button(buttons, text="Stop", command=self.stop_agent).pack(side="left", padx=6)
        ttk.Button(buttons, text="Clear Log", command=self.clear_log).pack(side="left")
        ttk.Button(buttons, text="Clear Memory", command=self.clear_memory).pack(side="left", padx=6)
        self.continue_btn = ttk.Button(
            buttons, text="Continue (PROMPT_USER)", command=self.continue_prompt_user, state="disabled"
        )
        self.continue_btn.pack(side="left", padx=6)

        log_frame = ttk.Frame(outer)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        input_row = ttk.Frame(outer)
        input_row.pack(fill="x", pady=(8, 0))
        self.user_input = ttk.Entry(input_row)
        self.user_input.pack(side="left", fill="x", expand=True)
        self.user_input.bind("<Return>", lambda _e: self.send_note())
        ttk.Button(input_row, text="Your response", command=self.send_note).pack(side="left", padx=(6, 0))

        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(8, 0))
        ttk.Label(
            outer,
            text="Cmds: memorize_thoughts · write_code · answer · talk_to_user · done · "
                 "(execute_python/shell/web/ingest gated by FILES_OFF)",
        ).pack(anchor="w", pady=(2, 0))

    def _collect_params(self) -> MiniAGIParams:
        self.params.update({
            "MODEL": self.var_model.get(),
            "SUMMARIZER_MODEL": self.var_summarizer.get(),
            "ENABLE_CRITIC": self.var_critic.get(),
            "PROMPT_USER": self.var_prompt_user.get(),
            "MAX_CONTEXT_SIZE": self.var_ctx.get(),
            "MAX_MEMORY_ITEM_SIZE": self.var_mem.get(),
            "SUMMARIZER_CHUNK_SIZE": self.var_chunk.get(),
            "WORK_DIR": self.var_workdir.get(),
            "DEBUG": self.var_debug.get(),
            "FILES_OFF": self.var_files_off.get(),
            "MAX_STEPS": self.var_steps.get(),
            "REASONING_EFFORT": self.var_effort.get(),
            "BITNET_HOME": self.var_home.get(),
            "BITNET_GGUF": self.var_gguf.get(),
            "BITNET_BACKEND": self.var_backend.get(),
            "BITNET_THREADS": self.var_threads.get(),
            "BITNET_TEMP": self.var_temp.get(),
            "BITNET_N_PREDICT": self.var_npred.get(),
        })
        self.params.apply_work_dir()
        self.var_workdir.set(self.params.work_dir)
        return self.params

    def save_params(self):
        p = self._collect_params()
        p.save_env()
        self.memory.data["miniagi_params"] = dict(p.values)
        self.memory.save()
        self.status.set(self._ready_status())
        self._append("SYSTEM", f"Saved MiniAGI parameters to {ENV_FILE} and persistent memory.")

    def _restore_memory(self):
        self._restoring = True
        saved = list(self.memory.data.get("log") or [])
        pending = []
        if self.memory.first_boot:
            msg = (
                f"First OS detect of {AGENT_NAME}. MiniAGI parameters loaded "
                f"(see Settings / .env_example). Memory: {self.memory.path}"
            )
            self._append("SYSTEM", msg, persist=False)
            pending.append(("SYSTEM", msg))
        elif saved:
            for e in saved:
                self._append(e.get("who", "SYSTEM"), e.get("text", ""), persist=False)
            msg = f"Restored {len(saved)} memory entries. WORK_DIR={self.params.work_dir}"
            self._append("SYSTEM", msg, persist=False)
            pending.append(("SYSTEM", msg))
        else:
            msg = f"{AGENT_NAME} ready with full MiniAGI parameter set."
            self._append("SYSTEM", msg, persist=False)
            pending.append(("SYSTEM", msg))
        self._restoring = False
        for who, text in pending:
            self.memory.append_log(who, text)

    def _append(self, who: str, text: str, persist: bool = True):
        body = text.strip()
        self.log.configure(state="normal")
        self.log.insert("end", f"\n[{who}]\n{body}\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        if persist and not self._restoring:
            self.memory.append_log(who, body)

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._append("SYSTEM", "Log view cleared.")

    def clear_memory(self):
        self.running = False
        self.agent = None
        self.memory.clear()
        self._restoring = True
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._restoring = False
        self.run_btn.configure(state="normal")
        self.continue_btn.configure(state="disabled")
        self.status.set(self._ready_status())
        self._append("SYSTEM", f"Persistent memory cleared at {self.memory.path}.")

    def start_agent(self):
        if self.running:
            return
        objective = self.objective.get("1.0", "end").strip()
        if not objective:
            self._append("SYSTEM", "Enter an objective first.")
            return
        params = self._collect_params()
        self.memory.data["objective"] = objective
        self.memory.data["miniagi_params"] = dict(params.values)
        self.memory.save()
        try:
            os.chdir(params.work_dir)
        except OSError as exc:
            self._append("SYSTEM", f"WORK_DIR error: {exc}")

        self.agent = CatSeekR1MiniAGI(objective, params)
        # Fresh session for coding objectives so stale ``` memory cannot skip write_code.
        if BitNetCoder.is_coding_request(objective):
            self.agent.restore_memory("", [])
            self.memory.data["summarized_history"] = ""
            self.memory.data["action_memory"] = []
            self.memory.save()
        else:
            self.agent.restore_memory(
                self.memory.data.get("summarized_history") or "",
                list(self.memory.data.get("action_memory") or []),
            )
        self.running = True
        self._awaiting_prompt_user = False
        self.run_btn.configure(state="disabled")
        self.continue_btn.configure(state="disabled")
        self.status.set(
            f"Running · MODEL={params.agent_model} · ctx={params.max_context_size} · "
            f"critic={'on' if params.enable_critic else 'off'}"
        )
        self._append("OBJECTIVE", objective)
        self._append(
            "SYSTEM",
            f"BitNet: {BitNetEngine(params).status()} · "
            f"MODEL={params.agent_model} SUMMARIZER={params.summarizer_model} "
            f"MAX_CONTEXT_SIZE={params.max_context_size} ENABLE_CRITIC={params.enable_critic} "
            f"PROMPT_USER={params.prompt_user} WORK_DIR={params.work_dir} "
            f"FILES_OFF={params.files_off} · code prints to terminal on write_code",
        )
        threading.Thread(target=self._agent_loop, daemon=True).start()

    def stop_agent(self):
        self.running = False
        self._awaiting_prompt_user = False
        self.continue_btn.configure(state="disabled")
        self._persist_agent()
        self.status.set(self._ready_status())
        self.run_btn.configure(state="normal")

    def continue_prompt_user(self):
        if self._awaiting_prompt_user:
            self.events.put(("continue", ""))
            self.continue_btn.configure(state="disabled")
            self._awaiting_prompt_user = False

    def send_note(self):
        note = self.user_input.get().strip()
        if not note:
            return
        self.user_input.delete(0, "end")
        self._append("YOU", note)
        if self.agent:
            self.events.put(("note", note))
            if self._awaiting_prompt_user:
                self.continue_btn.configure(state="disabled")
                self._awaiting_prompt_user = False

    def _persist_agent(self):
        if not self.agent:
            return
        summary, actions = self.agent.export_memory()
        self.memory.data["summarized_history"] = summary
        self.memory.data["action_memory"] = actions
        self.memory.save()

    def _agent_loop(self):
        pending_note = ""
        try:
            assert self.agent is not None
            while self.running and self.agent.step < self.agent.max_steps:
                try:
                    while True:
                        kind, value = self.events.get_nowait()
                        if kind == "note":
                            pending_note = value
                        elif kind == "continue":
                            pending_note = pending_note or ""
                except queue.Empty:
                    pass

                if pending_note:
                    self.agent.user_response(pending_note)
                    pending_note = ""

                self.agent.think()
                thought, command, arg_preview = self.agent.read_mind()
                self.events.put((
                    "mind",
                    f"{AGENT_NAME}: {thought}\nCmd: {command}, Arg: {arg_preview}",
                ))

                if self.agent.enable_critic:
                    criticism = self.agent.criticize()
                    if criticism:
                        self.events.put(("critic", criticism))

                # MiniAGI PROMPT_USER: pause before acting (except talk_to_user/done/memorize)
                if (
                    self.agent.prompt_user
                    and self.agent.pending_user_gate
                    and command not in {"talk_to_user", "done", "memorize_thoughts"}
                ):
                    self.events.put((
                        "prompt_user",
                        "PROMPT_USER=true — press Continue or type feedback before this action runs.",
                    ))
                    # Wait for continue or note
                    while self.running:
                        try:
                            kind, value = self.events.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        if kind == "continue":
                            break
                        if kind == "note":
                            self.agent.user_response(value)
                            self.events.put(("user_feedback", value))
                            # abort this action like MiniAGI when feedback given
                            break
                        # re-queue other events
                        self.events.put((kind, value))
                        break
                    else:
                        break
                    # If user sent feedback, skip act this round
                    if self.agent.last_observation == pending_note:
                        continue

                result = self.agent.act()
                if result.get("command") == "write_code":
                    self.events.put(("code", result.get("observation") or ""))
                self.events.put(("result", result))
                if result["action"] in {"ASK", "DONE"}:
                    break

            self.events.put(("finished", None))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("error", str(exc)))

    def _poll_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "mind":
                    self._append("MIND", value)
                elif kind == "critic":
                    self._append("CRITIC", value)
                elif kind == "prompt_user":
                    self._append("PROMPT_USER", value)
                    self._awaiting_prompt_user = True
                    self.continue_btn.configure(state="normal")
                elif kind == "user_feedback":
                    self._append("FEEDBACK", str(value))
                elif kind == "code":
                    self._append("CODE", value)
                elif kind == "result":
                    self._append(str(value.get("command", "cmd")).upper(), value.get("observation") or "")
                    self._persist_agent()
                elif kind == "error":
                    self._append("ERROR", value)
                    self.running = False
                    self.run_btn.configure(state="normal")
                    self.continue_btn.configure(state="disabled")
                elif kind == "finished":
                    self.running = False
                    self.run_btn.configure(state="normal")
                    self.continue_btn.configure(state="disabled")
                    self._persist_agent()
                    self.status.set(self._ready_status())
                elif kind in {"note", "continue"}:
                    self.events.put((kind, value))
                    break
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _on_close(self):
        try:
            self._collect_params()
            self.memory.data["objective"] = self.objective.get("1.0", "end").strip()
            self.memory.data["miniagi_params"] = dict(self.params.values)
            self._persist_agent()
            self.memory.save()
        except OSError:
            pass
        self.destroy()


def main(argv: list[str] | None = None):
    """CLI mirrors microsoft/BitNet run_inference.py plus MiniAGI GUI.

    BitNet-compatible:
      -m/--model  -n/--n-predict  -p/--prompt  -t/--threads
      -c/--ctx-size  -temp/--temperature  -cnv/--conversation

    CatSeek extras:
      --setup-bitnet  --bitnet-status  (no flags → GUI)
    """
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="CatSeek R1 — real BitNet b1.58 MiniAGI (microsoft/BitNet)",
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to GGUF (BitNet -m)")
    parser.add_argument("-n", "--n-predict", type=int, default=None, help="Tokens to predict (BitNet -n)")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Prompt / system prompt (BitNet -p)")
    parser.add_argument("-t", "--threads", type=int, default=None, help="CPU threads (BitNet -t)")
    parser.add_argument("-c", "--ctx-size", type=int, default=None, help="Context size (BitNet -c)")
    parser.add_argument("-temp", "--temperature", type=float, default=None, help="Temperature (BitNet -temp)")
    parser.add_argument("-cnv", "--conversation", action="store_true", help="Chat mode (BitNet -cnv)")
    parser.add_argument("--setup-bitnet", action="store_true", help="Clone/build microsoft/BitNet + download GGUF")
    parser.add_argument("--bitnet-status", action="store_true", help="Print BitNet backend status and exit")
    parser.add_argument("--gui", action="store_true", help="Force MiniAGI GUI")
    parser.add_argument("objective", nargs="*", help="MiniAGI objective (GUI mode)")
    args, unknown = parser.parse_known_args(argv)

    write_env_example()
    params = MiniAGIParams.from_env_file()
    if args.model:
        params.values["BITNET_GGUF"] = args.model
        params.values["BITNET_BACKEND"] = "cpp"
    if args.n_predict is not None:
        params.values["BITNET_N_PREDICT"] = args.n_predict
    if args.threads is not None:
        params.values["BITNET_THREADS"] = args.threads
    if args.ctx_size is not None:
        params.values["MAX_CONTEXT_SIZE"] = args.ctx_size
    if args.temperature is not None:
        params.values["BITNET_TEMP"] = args.temperature

    if args.setup_bitnet:
        print(setup_bitnet(params))
        return

    if args.bitnet_status:
        print(BitNetEngine(params).status())
        print(f"BitLinear bits/weight = {BitLinear.bits_per_weight():.5f}")
        return

    # Direct BitNet inference path (exactly like run_inference.py)
    if args.prompt is not None and not args.gui:
        engine = BitNetEngine(params)
        if not engine.is_live:
            print("ERROR: " + engine.status(), file=sys.stderr)
            print('Run with --setup-bitnet first.', file=sys.stderr)
            sys.exit(1)
        out = engine.generate(
            args.prompt if not unknown else args.prompt,
            n_predict=params.bitnet_n_predict,
            system=None,
            conversation=bool(args.conversation),
        )
        # If extra positional text was given with -cnv, treat as user turn.
        if args.conversation and (args.objective or unknown):
            user = " ".join(list(args.objective) + list(unknown)).strip()
            if user:
                out = engine.generate(
                    user,
                    n_predict=params.bitnet_n_predict,
                    system=args.prompt,
                    conversation=True,
                )
        print(out)
        return

    objective = " ".join(args.objective).strip() or None
    if unknown and not objective:
        objective = " ".join(unknown).strip() or None
    CatSeekGUI(initial_objective=objective).mainloop()


if __name__ == "__main__":
    main()
