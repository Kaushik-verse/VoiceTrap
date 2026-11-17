"""Minimal Groq client (free tier): Whisper transcription, chat with tool calling, Prompt Guard.

* Retries on 429/5xx and honours ``retry-after``.
* Rotates to the next API key (``GROQ_API_KEY``, ``GROQ_API_KEY_2``, ...) when a key's daily quota is used up.
* Caches every response on disk by a hash of the request, so benchmark runs are resumable and
  re-scoring never spends quota twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

BASE = "https://api.groq.com/openai/v1"
ASR_MODEL = os.getenv("VOICETRAP_ASR_MODEL", "whisper-large-v3-turbo")
AGENT_MODEL = os.getenv("VOICETRAP_AGENT_MODEL", "openai/gpt-oss-20b")
GUARD_MODEL = os.getenv("VOICETRAP_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")


class GroqError(RuntimeError):
    pass


def _keys() -> list[str]:
    keys = [os.getenv("GROQ_API_KEY", "")] + [os.getenv(f"GROQ_API_KEY_{i}", "") for i in range(2, 6)]
    return [k for k in keys if k]


class Groq:
    def __init__(self, cache_dir: str | Path = ".cache/groq", keys: list[str] | None = None,
                 max_retries: int = 6) -> None:
        self.keys = keys if keys is not None else _keys()
        self.key_idx = 0
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.http = httpx.Client(timeout=90, headers={"User-Agent": "voicetrap/0.1"})
        self.calls = 0
        self.cache_hits = 0
        self.tokens = 0

    # ---------- plumbing ----------
    def _cached(self, kind: str, payload: dict, fn):
        key = hashlib.sha256(json.dumps([kind, payload], sort_keys=True, default=str).encode()).hexdigest()[:24]
        path = self.cache / f"{kind}-{key}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())
        out = fn()
        path.write_text(json.dumps(out))
        return out

    def _post(self, url: str, **kw) -> dict:
        if not self.keys:
            raise GroqError("No GROQ_API_KEY set. Get a free key at https://console.groq.com/keys")
        for attempt in range(self.max_retries):
            try:
                r = self.http.post(url, headers={"Authorization": f"Bearer {self.keys[self.key_idx]}"}, **kw)
            except httpx.TransportError:  # network blip (DNS, reset, timeout): back off and retry
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            self.calls += 1
            if r.status_code == 200:
                return r.json()
            body = r.text[:400]
            if r.status_code == 429 and ("per day" in body or "(TPD)" in body or "(RPD)" in body or
                                         "(ASD)" in body):
                if self.key_idx + 1 < len(self.keys):
                    self.key_idx += 1  # this key's daily quota is used up: rotate
                    continue
                raise GroqError(f"daily free-tier quota exhausted on all keys: {body}")
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("retry-after", 0) or 0) or min(60, 2 ** attempt * 2)
                time.sleep(min(wait, 90))
                continue
            raise GroqError(f"HTTP {r.status_code}: {body}")
        raise GroqError(f"gave up after {self.max_retries} retries: {url}")

    # ---------- endpoints ----------
    def transcribe(self, wav: bytes, model: str = ASR_MODEL) -> dict:
        """Returns ``{"text": ..., "segments": [{start, end, text, no_speech_prob}]}``."""
        digest = hashlib.sha256(wav).hexdigest()

        def call() -> dict:
            d = self._post(f"{BASE}/audio/transcriptions",
                           files={"file": ("audio.wav", wav, "audio/wav")},
                           data={"model": model, "response_format": "verbose_json", "language": "en",
                                 "temperature": "0"})
            segs = [{k: s.get(k) for k in ("start", "end", "text", "no_speech_prob")}
                    for s in d.get("segments", [])]
            return {"text": (d.get("text") or "").strip(), "segments": segs}

        return self._cached("asr", {"audio": digest, "model": model}, call)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, model: str = AGENT_MODEL,
             temperature: float = 0.0) -> dict:
        payload = {"model": model, "messages": messages, "temperature": temperature}
        if tools:
            payload |= {"tools": tools, "tool_choice": "auto"}
        if "gpt-oss" in model:
            payload["reasoning_effort"] = os.getenv("VOICETRAP_REASONING_EFFORT", "low")

        def call() -> dict:
            d = self._post(f"{BASE}/chat/completions", json=payload)
            msg = d["choices"][0]["message"]
            raw = msg.get("tool_calls") or []
            calls = [{"id": c.get("id"), "name": c["function"]["name"],
                      "arguments": _args(c["function"].get("arguments"))} for c in raw]
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if raw:
                assistant["tool_calls"] = [{"id": c.get("id"), "type": "function",
                                            "function": {"name": c["function"]["name"],
                                                         "arguments": c["function"].get("arguments") or "{}"}}
                                           for c in raw]
            return {"content": msg.get("content") or "", "tool_calls": calls, "message": assistant,
                    "tokens": d.get("usage", {}).get("total_tokens", 0)}

        out = self._cached("chat", payload, call)
        self.tokens += out.get("tokens", 0)
        return out

    def prompt_guard(self, text: str, model: str = GUARD_MODEL) -> float:
        """Probability that ``text`` is a prompt injection / jailbreak (Llama Prompt Guard 2)."""
        out = self._cached("guard", {"model": model, "text": text}, lambda: self._post(
            f"{BASE}/chat/completions", json={"model": model, "messages": [{"role": "user", "content": text}]}))
        try:
            return float(out["choices"][0]["message"]["content"])
        except (KeyError, ValueError, TypeError):
            return 0.0


def _args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw}
