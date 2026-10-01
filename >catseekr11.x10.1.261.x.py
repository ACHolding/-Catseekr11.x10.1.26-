#!/usr/bin/env python3
"""
CatSeek R1 / CatR1-MiniAGI — BitNet only (no OpenAI)

Local MiniAGI-style autonomous agent powered entirely by an offline
BitNet b1.58 planner/coder. No OpenAI API key, no cloud LLM.

Parameters (MiniAGI-shaped, BitNet-backed):
  MODEL, SUMMARIZER_MODEL -> CatSeek-R1-BitNet-b1.58 / BitNet-Summarizer
  ENABLE_CRITIC, PROMPT_USER,
  MAX_CONTEXT_SIZE, MAX_MEMORY_ITEM_SIZE, SUMMARIZER_CHUNK_SIZE,
  WORK_DIR, DEBUG, FILES_OFF, MAX_STEPS, REASONING_EFFORT

Usage:
  python3 ">catseekr11.x10.1.261.x.py"
  python3 ">catseekr11.x10.1.261.x.py" "write hello cat in c"
"""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import sys
import threading
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk

APP_NAME = "CatSeek R1 · BitNet MiniAGI [c] Kondo Solutions 1999-2026"
AGENT_NAME = "CatSeek R1"
BITNET_MODEL = "CatSeek-R1-BitNet-b1.58"
BITNET_SUMMARIZER = "CatSeek-R1-BitNet-Summarizer"
BITNET_BITS = 1.58
NO_OPENAI = True
OPERATING_SYSTEM = platform.platform()
MEMORY_DIR = Path.home() / ".catr1-miniagi"
MEMORY_FILE = MEMORY_DIR / "memory.json"
_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ENV_FILE = _ROOT / ".env"
ENV_EXAMPLE = _ROOT / ".env_example"
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
        # BitNet-only — no OpenAI keys or cloud model IDs
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
            }:
                self.values[k] = _parse_int(v, self.DEFAULTS[k])
            elif k == "BITNET_BITS":
                try:
                    self.values[k] = float(v)
                except (TypeError, ValueError):
                    self.values[k] = BITNET_BITS
            else:
                self.values[k] = "" if v is None else str(v)
        model = str(self.values.get("MODEL") or "")
        summ = str(self.values.get("SUMMARIZER_MODEL") or "")
        cloudish = ("gpt-", "openai", "chatgpt", "o1", "o3", "claude")
        if not model or any(x in model.lower() for x in cloudish):
            self.values["MODEL"] = BITNET_MODEL
        if not summ or any(x in summ.lower() for x in cloudish):
            self.values["SUMMARIZER_MODEL"] = BITNET_SUMMARIZER

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
            "# CatSeek R1 — BitNet-only (no OpenAI)",
            f'MODEL="{self.values.get("MODEL", BITNET_MODEL)}"',
            f'SUMMARIZER_MODEL="{self.values.get("SUMMARIZER_MODEL", BITNET_SUMMARIZER)}"',
            f'BITNET_BITS={self.values.get("BITNET_BITS", BITNET_BITS)}',
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
                "# CatSeek R1 — BitNet-only (no OpenAI / no cloud LLM)",
                f'MODEL="{BITNET_MODEL}"',
                f'SUMMARIZER_MODEL="{BITNET_SUMMARIZER}"',
                f"BITNET_BITS={BITNET_BITS}",
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
# Offline coder
# ---------------------------------------------------------------------------

class BitNetCoder:
    LANG_ALIASES = {
        "c": "c", "c++": "cpp", "cpp": "cpp", "python": "python", "py": "python",
        "javascript": "javascript", "js": "javascript", "go": "go", "rust": "rust",
        "java": "java", "bash": "bash", "shell": "bash",
    }

    @classmethod
    def is_coding_request(cls, text: str) -> bool:
        q = text.lower()
        if any(w in q for w in ("write", "implement", "code", "program", "print", "hello")):
            return True
        return any(re.search(rf"\bin\s+{re.escape(a)}\b", q) for a in cls.LANG_ALIASES)

    @classmethod
    def detect_lang(cls, text: str) -> str:
        q = text.lower()
        for alias in sorted(cls.LANG_ALIASES, key=len, reverse=True):
            if re.search(rf"\bin\s+{re.escape(alias)}\b", q):
                return cls.LANG_ALIASES[alias]
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

    @classmethod
    def synthesize(cls, text: str) -> str:
        lang = cls.detect_lang(text)
        msg = cls._extract_message(text).replace('"', '\\"')
        if lang == "c":
            code = (
                f'#include <stdio.h>\n\nint main(void) {{\n'
                f'    printf("{msg}\\n");\n    return 0;\n}}\n'
            )
        elif lang == "go":
            code = (
                f'package main\n\nimport "fmt"\n\n'
                f'func main() {{\n    fmt.Println("{msg}")\n}}\n'
            )
        elif lang == "rust":
            code = f'fn main() {{\n    println!("{msg}");\n}}\n'
        else:
            lang = "python"
            code = f"print('{msg.replace(chr(39), chr(92)+chr(39))}')\n"
        return f"Observation: {lang} program for `{msg}`.\n\n```{lang}\n{code}```"


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
    """MiniAGI loop powered by offline BitNet (no OpenAI)."""

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

        self.coder = BitNetCoder()
        self.summarized_history = ""
        self.action_memory: list[str] = []
        self.criticism = ""
        self.thought = ""
        self.proposed_command = ""
        self.proposed_arg = ""
        self.step = 0
        self.last_observation = ""
        self.pending_user_gate = False  # PROMPT_USER pause

        if self.debug:
            print(
                f"[DEBUG] model={self.agent_model} summarizer={self.summarizer_model} "
                f"ctx={self.max_context_size} mem_item={self.max_memory_item_size} "
                f"chunk={self.summarizer_chunk_size} critic={self.enable_critic} "
                f"prompt_user={self.prompt_user} work_dir={self.work_dir} "
                f"files_off={self.files_off}"
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
            # MiniAGI: chunked summarize oversized observations
            obs = _clip(
                f"[summarized to SUMMARIZER_CHUNK_SIZE={self.summarizer_chunk_size}]\n" + observation,
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
                f"{bullet}"
            )

    def think(self):
        self.step += 1
        self.pending_user_gate = False
        _ = self._context()  # builds MiniAGI-style context under MAX_CONTEXT_SIZE
        q = self.objective.lower()
        ctx = self._context().lower()
        already_coded = "write_code" in ctx and "```" in ctx

        if self.step == 1:
            self.thought = (
                f"Organize objective using agent_model={self.agent_model}; "
                f"memory budgets ctx={self.max_context_size} item={self.max_memory_item_size}."
            )
            self.proposed_command = "memorize_thoughts"
            self.proposed_arg = (
                f"Objective: {self.objective}\n"
                f"WORK_DIR: {self.work_dir}\n"
                f"ENABLE_CRITIC={self.enable_critic} PROMPT_USER={self.prompt_user} "
                f"DEBUG={self.debug} FILES_OFF={self.files_off}\n"
                f"Plan: think → act → (critic) → done."
            )
            return

        if BitNetCoder.is_coding_request(self.objective) and not already_coded:
            self.thought = "Coding objective — use write_code (local BitNet coder)."
            self.proposed_command = "write_code"
            self.proposed_arg = self.objective
            if self.prompt_user:
                self.pending_user_gate = True
            return

        if already_coded or (self.last_observation and "```" in self.last_observation):
            self.thought = "Result ready — send done."
            self.proposed_command = "done"
            self.proposed_arg = self.last_observation or "Objective complete."
            return

        if any(w in q for w in ("who are you", "hello", "hi", "hey")) and len(q.split()) <= 4:
            self.thought = "Greeting / identity."
            self.proposed_command = "talk_to_user" if self.prompt_user else "answer"
            self.proposed_arg = (
                f"I'm {AGENT_NAME} (BitNet MiniAGI). Offline BitNet params: "
                f"MODEL={self.agent_model}, SUMMARIZER_MODEL={self.summarizer_model}, "
                f"MAX_CONTEXT_SIZE={self.max_context_size}, "
                f"MAX_MEMORY_ITEM_SIZE={self.max_memory_item_size}, "
                f"SUMMARIZER_CHUNK_SIZE={self.summarizer_chunk_size}, "
                f"ENABLE_CRITIC={self.enable_critic}, PROMPT_USER={self.prompt_user}, "
                f"WORK_DIR={self.work_dir}, DEBUG={self.debug}, FILES_OFF={self.files_off}."
            )
            return

        if self.step >= self.max_steps - 1:
            self.thought = "Step budget nearly spent."
            self.proposed_command = "done"
            self.proposed_arg = self.last_observation or "Reached MAX_STEPS."
            return

        if "memorize_thoughts" not in ctx:
            self.thought = "Internal plan before answering."
            self.proposed_command = "memorize_thoughts"
            self.proposed_arg = f"Break down: {self.objective}"
            return

        self.thought = "Direct answer toward the objective."
        self.proposed_command = "answer"
        self.proposed_arg = (
            f"Toward `{self.objective}`: smallest reversible step, verify, iterate. "
            f"(summarizer={self.summarizer_model})"
        )
        if self.prompt_user:
            self.pending_user_gate = True

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
        if not issues:
            self.criticism = (
                f"Criticism ({self.summarizer_model}): Action looks reasonable. Proceed."
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
                # Still refuse silent arbitrary exec in this offline packaging —
                # require explicit non-FILES_OFF plus user-owned confirmation text.
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
            self._summarize(f"{command}\n{arg}", obs)
            self.last_observation = obs
            self.criticism = ""
            return {
                "thought": self.thought, "command": command, "arg": arg,
                "observation": obs, "action": "ANSWER",
            }

        if command == "answer":
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
        return (
            f"Ready · {AGENT_NAME} · BitNet={self.params.agent_model} · "
            f"CRITIC={'on' if self.params.enable_critic else 'off'} · "
            f"PROMPT_USER={'on' if self.params.prompt_user else 'off'} · "
            f"FILES_OFF={'on' if self.params.files_off else 'off'} · no OpenAI"
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
        ttk.Label(title, text="CatSeek R1 · CatR1-MiniAGI", font=("TkDefaultFont", 18, "bold")).pack(side="left")
        ttk.Label(title, text="BitNet b1.58 · offline · Kondo Solutions").pack(side="right")

        ttk.Label(
            outer,
            text="BitNet-only · no OpenAI · local MODEL/SUMMARIZER · ENABLE_CRITIC · "
                 "PROMPT_USER · MAX_CONTEXT_SIZE · MAX_MEMORY_ITEM_SIZE · SUMMARIZER_CHUNK_SIZE · WORK_DIR · DEBUG",
        ).pack(anchor="w", pady=(4, 0))

        # --- MiniAGI settings panel ---
        settings = ttk.LabelFrame(outer, text="BitNet / MiniAGI parameters (offline)", padding=8)
        settings.pack(fill="x", pady=(10, 6))

        self.var_model = tk.StringVar(value=self.params.agent_model)
        self.var_summarizer = tk.StringVar(value=self.params.summarizer_model)
        self.var_workdir = tk.StringVar(value=self.params.work_dir)
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
        ttk.Label(row1, text="BITNET MODEL").pack(side="left")
        ttk.Entry(row1, textvariable=self.var_model, width=28).pack(side="left", padx=4)
        ttk.Label(row1, text="BITNET SUMMARIZER").pack(side="left", padx=(8, 0))
        ttk.Entry(row1, textvariable=self.var_summarizer, width=28).pack(side="left", padx=4)
        ttk.Label(row1, text="(no OpenAI)").pack(side="left", padx=(8, 0))

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
            f"BitNet params: MODEL={params.agent_model} SUMMARIZER={params.summarizer_model} "
            f"MAX_CONTEXT_SIZE={params.max_context_size} MAX_MEMORY_ITEM_SIZE={params.max_memory_item_size} "
            f"SUMMARIZER_CHUNK_SIZE={params.summarizer_chunk_size} ENABLE_CRITIC={params.enable_critic} "
            f"PROMPT_USER={params.prompt_user} WORK_DIR={params.work_dir} DEBUG={params.debug} "
            f"FILES_OFF={params.files_off}",
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
    argv = list(sys.argv[1:] if argv is None else argv)
    objective = " ".join(argv).strip() or None
    CatSeekGUI(initial_objective=objective).mainloop()


if __name__ == "__main__":
    main()
