"""Context window monitor for AgentAutopsy.

Tracks token usage against model context limits in real time, warns before
the window fills up, detects silent truncation, and suggests optimizations.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from agentautopsy.db import create_tables, get_db, insert_event, insert_run

# ── Model context limits (tokens) ────────────────────────────────────────────
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o":             128_000,
    "gpt-4o-mini":        128_000,
    "claude-opus-4-6":    200_000,
    "claude-sonnet-4-6":  200_000,
    "claude-haiku-4-5":   200_000,
    "gemini-1.5-pro":   1_000_000,
    "gemini-1.5-flash":   1_000_000,
    "llama-3":            128_000,
    # common aliases
    "gpt-4":               8_192,
    "gpt-4-32k":          32_768,
    "gpt-3.5-turbo":      16_385,
    "claude-3-opus":      200_000,
    "claude-3-sonnet":    200_000,
    "claude-3-haiku":     200_000,
    "claude-2":           100_000,
}

_DEFAULT_CONTEXT_LIMIT = 128_000

_context_monitor_ctx: dict[str, Any] = {}

# Threshold fractions
WARN_FRACTION = 0.70
CRITICAL_FRACTION = 0.90

# Role labels for the breakdown
_ROLE_LABELS: dict[str, str] = {
    "system":    "System prompt",
    "user":      "User message",
    "assistant": "Assistant response",
    "tool":      "Tool response",
    "function":  "Function response",
}


# ── Token counting ────────────────────────────────────────────────────────────

def _count_tokens_rough(text: str) -> int:
    """Rough token estimate: ~4 chars per token (no tiktoken dependency)."""
    return max(1, len(text) // 4)


def _tokens_in_message(msg: Any) -> int:
    """Count tokens in a single message dict or string."""
    if isinstance(msg, str):
        return _count_tokens_rough(msg)
    if isinstance(msg, dict):
        total = 0
        for key in ("content", "text", "value"):
            val = msg.get(key)
            if isinstance(val, str):
                total += _count_tokens_rough(val)
            elif isinstance(val, list):
                for part in val:
                    total += _tokens_in_message(part)
        return total or _count_tokens_rough(json.dumps(msg, default=str))
    return _count_tokens_rough(str(msg))


def _tokens_in_payload(payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    """Return (total_tokens, per_message_breakdown) from an LLM call payload."""
    messages_raw = payload.get("messages") or []
    breakdown: list[dict[str, Any]] = []

    if isinstance(messages_raw, list):
        for idx, msg in enumerate(messages_raw):
            role = "unknown"
            if isinstance(msg, dict):
                role = str(msg.get("role") or "unknown")
            tok = _tokens_in_message(msg)
            breakdown.append({
                "index": idx,
                "role": role,
                "label": _ROLE_LABELS.get(role, role),
                "tokens": tok,
            })
    elif isinstance(messages_raw, str):
        tok = _count_tokens_rough(messages_raw)
        breakdown.append({"index": 0, "role": "user", "label": "Prompt", "tokens": tok})

    # Also count prompt / system fields if present (Anthropic-style)
    for field in ("prompt", "system"):
        val = payload.get(field)
        if isinstance(val, str) and val:
            tok = _count_tokens_rough(val)
            breakdown.append({"index": -1, "role": field, "label": _ROLE_LABELS.get(field, field), "tokens": tok})

    total = sum(b["tokens"] for b in breakdown)
    return total, breakdown


def _resolve_context_limit(model: str) -> int:
    key = model.lower().strip()
    limit = MODEL_CONTEXT_LIMITS.get(key)
    if limit is None:
        for prefix, lim in MODEL_CONTEXT_LIMITS.items():
            if key.startswith(prefix) or prefix.startswith(key):
                limit = lim
                break
    return limit or _DEFAULT_CONTEXT_LIMIT


# ── DB schema ─────────────────────────────────────────────────────────────────

def ensure_context_tables(db: Any) -> None:
    """Create context_snapshots table if it does not exist."""
    db["context_snapshots"].create(
        {
            "id": str,
            "run_id": str,
            "step": int,
            "recorded_at": str,
            "model": str,
            "context_limit": int,
            "tokens_used": int,
            "pct_used": float,
            "alert_level": str,
            "breakdown_json": str,
            "suggestions_json": str,
            "truncation_suspected": int,
        },
        pk="id",
        if_not_exists=True,
    )


# ── Optimization suggestions ──────────────────────────────────────────────────

def _generate_suggestions(
    breakdown: list[dict[str, Any]],
    tokens_used: int,
    context_limit: int,
) -> list[str]:
    suggestions: list[str] = []
    pct = tokens_used / context_limit if context_limit else 0

    if not breakdown:
        return suggestions

    total = max(tokens_used, 1)

    # Rank messages by token count
    sorted_msgs = sorted(breakdown, key=lambda m: m["tokens"], reverse=True)
    top = sorted_msgs[0]
    top_pct = int(top["tokens"] / total * 100)

    if top["tokens"] > 8_000:
        label = top.get("label") or top.get("role") or "message"
        suggestions.append(
            f'Step {top["index"] + 1} {label} is {top["tokens"]} tokens '
            f"({top_pct}% of context) — consider summarizing it"
        )

    # System prompt warning
    sys_msgs = [m for m in breakdown if m["role"] == "system"]
    if sys_msgs:
        sys_tok = sum(m["tokens"] for m in sys_msgs)
        sys_pct = int(sys_tok / context_limit * 100)
        if sys_pct >= 20:
            suggestions.append(
                f"System prompt is {sys_tok} tokens — {sys_pct}% of your context budget"
            )

    # Tool response bloat
    tool_msgs = [m for m in breakdown if m["role"] in ("tool", "function")]
    if tool_msgs:
        tool_tok = sum(m["tokens"] for m in tool_msgs)
        if tool_tok > 5_000:
            suggestions.append(
                f"Tool responses total {tool_tok} tokens — consider filtering or truncating outputs"
            )

    # History growth
    user_msgs = [m for m in breakdown if m["role"] == "user"]
    if len(user_msgs) >= 5:
        suggestions.append(
            f"Conversation has {len(user_msgs)} user turns — "
            "consider windowing to the most recent N messages"
        )

    if pct >= CRITICAL_FRACTION:
        # Rank top-3 space hogs
        for msg in sorted_msgs[:3]:
            msg_pct = int(msg["tokens"] / total * 100)
            label = msg.get("label") or msg.get("role") or "message"
            suggestions.append(
                f'Context is nearly full: step {msg["index"] + 1} {label} '
                f"uses {msg_pct}% ({msg['tokens']} tokens)"
            )

    return suggestions


# ── Truncation detection ──────────────────────────────────────────────────────

def _detect_truncation(
    prev_output_tokens: int | None,
    curr_output_tokens: int | None,
) -> bool:
    """Heuristic: output shrank by more than 40% compared to previous call."""
    if prev_output_tokens is None or curr_output_tokens is None:
        return False
    if prev_output_tokens <= 10:
        return False
    ratio = curr_output_tokens / prev_output_tokens
    return ratio < 0.60


# ── ContextMonitor class ──────────────────────────────────────────────────────

class ContextMonitor:
    """Track context window usage per LLM call and alert before overflow.

    Parameters
    ----------
    warn_fraction:
        Issue a warning when context reaches this fraction (default 0.70 = 70%).
    critical_fraction:
        Issue a critical alert at this fraction (default 0.90 = 90%).
    """

    def __init__(
        self,
        db: Any | None = None,
        run_id: str | None = None,
        *,
        agent_name: str = "agent",
        warn_fraction: float = WARN_FRACTION,
        critical_fraction: float = CRITICAL_FRACTION,
    ) -> None:
        self.db = db
        self.run_id = run_id
        self.agent_name = agent_name
        self.warn_fraction = warn_fraction
        self.critical_fraction = critical_fraction

        self._lock = threading.Lock()
        self._step = 0
        self._snapshots: list[dict[str, Any]] = []
        self._prev_output_tokens: int | None = None
        self._warned: set[str] = set()  # "warn" / "critical" per run

    # ── Public API ────────────────────────────────────────────────────────────

    def watch(self) -> ContextMonitor:
        """Register as the active monitor so watch() auto-enables it."""
        if self.db is None:
            self.db = get_db()
        create_tables(self.db)
        ensure_context_tables(self.db)
        if self.run_id is None:
            self.run_id = insert_run(self.db, agent_name=self.agent_name)
        _context_monitor_ctx["monitor"] = self
        return self

    def record_llm_call(
        self,
        payload: dict[str, Any],
        *,
        model: str = "gpt-4o",
        output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Analyse a single LLM call payload.  Returns a status dict."""
        with self._lock:
            self._step += 1
            step = self._step

        tokens_used, breakdown = _tokens_in_payload(payload)

        # If the call provides token_input explicitly, trust that value
        explicit = payload.get("token_input") or payload.get("input_tokens")
        if isinstance(explicit, int) and explicit > 0:
            tokens_used = explicit

        context_limit = _resolve_context_limit(model)
        pct = tokens_used / context_limit if context_limit else 0

        truncation = _detect_truncation(self._prev_output_tokens, output_tokens)
        if output_tokens is not None:
            self._prev_output_tokens = output_tokens

        alert_level = "ok"
        if pct >= self.critical_fraction:
            alert_level = "critical"
        elif pct >= self.warn_fraction:
            alert_level = "warn"

        suggestions = _generate_suggestions(breakdown, tokens_used, context_limit)

        snap: dict[str, Any] = {
            "step": step,
            "model": model,
            "context_limit": context_limit,
            "tokens_used": tokens_used,
            "pct_used": round(pct * 100, 4),
            "alert_level": alert_level,
            "breakdown": breakdown,
            "suggestions": suggestions,
            "truncation_suspected": truncation,
        }

        with self._lock:
            self._snapshots.append(snap)

        self._persist(snap)
        self._maybe_insert_event(snap, alert_level, truncation)
        self._print_alert(snap, alert_level, truncation)

        return {
            "ok": alert_level == "ok",
            "alert_level": alert_level,
            "pct_used": snap["pct_used"],
            "tokens_used": tokens_used,
            "context_limit": context_limit,
            "suggestions": suggestions,
            "truncation_suspected": truncation,
        }

    def get_usage(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Return per-step context usage for this run (or load from DB)."""
        rid = run_id or self.run_id
        if rid and rid != self.run_id:
            return load_context_snapshots(self.db or get_db(), rid)
        with self._lock:
            return list(self._snapshots)

    def current_pct(self) -> float:
        """Latest context usage percentage (0-100)."""
        with self._lock:
            if not self._snapshots:
                return 0.0
            return self._snapshots[-1]["pct_used"]

    def reset(self) -> None:
        """Clear all counters for a new run."""
        with self._lock:
            self._step = 0
            self._snapshots.clear()
            self._prev_output_tokens = None
            self._warned.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _persist(self, snap: dict[str, Any]) -> None:
        if self.db is None or self.run_id is None:
            return
        import uuid
        try:
            ensure_context_tables(self.db)
            self.db["context_snapshots"].insert(
                {
                    "id": str(uuid.uuid4()),
                    "run_id": self.run_id,
                    "step": snap["step"],
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "model": snap["model"],
                    "context_limit": snap["context_limit"],
                    "tokens_used": snap["tokens_used"],
                    "pct_used": snap["pct_used"],
                    "alert_level": snap["alert_level"],
                    "breakdown_json": json.dumps(snap["breakdown"]),
                    "suggestions_json": json.dumps(snap["suggestions"]),
                    "truncation_suspected": int(snap["truncation_suspected"]),
                },
                pk="id",
            )
        except Exception:  # noqa: BLE001
            pass

    def _maybe_insert_event(
        self,
        snap: dict[str, Any],
        alert_level: str,
        truncation: bool,
    ) -> None:
        if self.db is None or self.run_id is None:
            return
        try:
            if alert_level in ("warn", "critical"):
                key = f"{alert_level}_{snap['step']}"
                if key not in self._warned:
                    self._warned.add(key)
                    insert_event(
                        self.db,
                        self.run_id,
                        "context_alert",
                        {
                            "alert_level": alert_level,
                            "step": snap["step"],
                            "pct_used": snap["pct_used"],
                            "tokens_used": snap["tokens_used"],
                            "context_limit": snap["context_limit"],
                            "model": snap["model"],
                            "suggestions": snap["suggestions"],
                        },
                    )
            if truncation:
                insert_event(
                    self.db,
                    self.run_id,
                    "context_truncation_suspected",
                    {
                        "step": snap["step"],
                        "pct_used": snap["pct_used"],
                        "prev_output_tokens": self._prev_output_tokens,
                    },
                )
        except Exception:  # noqa: BLE001
            pass

    def _print_alert(
        self,
        snap: dict[str, Any],
        alert_level: str,
        truncation: bool,
    ) -> None:
        pct = snap["pct_used"]
        model = snap["model"]
        tokens = snap["tokens_used"]
        limit = snap["context_limit"]

        if alert_level == "critical":
            print(
                f"\n\033[1;38;5;196m⚠ [AgentAutopsy] Context Critical "
                f"({pct:.1f}% — {tokens}/{limit} tokens) [{model}]\033[0m"
            )
            for sug in snap["suggestions"][:3]:
                print(f"\033[38;5;196m  → {sug}\033[0m")
        elif alert_level == "warn":
            print(
                f"\n\033[38;5;226m⚠ [AgentAutopsy] Context Warning "
                f"({pct:.1f}% — {tokens}/{limit} tokens) [{model}]\033[0m"
            )
            for sug in snap["suggestions"][:2]:
                print(f"\033[38;5;226m  → {sug}\033[0m")

        if truncation:
            print(
                f"\033[38;5;208m⚠ [AgentAutopsy] Possible silent truncation at step {snap['step']}\033[0m"
            )


# ── Module-level helpers ──────────────────────────────────────────────────────

def get_active_monitor() -> ContextMonitor | None:
    monitor = _context_monitor_ctx.get("monitor")
    return monitor if isinstance(monitor, ContextMonitor) else None


def record_llm_call_event(
    payload: dict[str, Any],
    *,
    model: str = "gpt-4o",
    output_tokens: int | None = None,
) -> dict[str, Any]:
    """Feed an LLM payload to the active ContextMonitor if one is registered."""
    monitor = get_active_monitor()
    if monitor is None:
        return {"ok": True}
    return monitor.record_llm_call(payload, model=model, output_tokens=output_tokens)


def load_context_snapshots(db: Any, run_id: str) -> list[dict[str, Any]]:
    """Load persisted context snapshots for a run from the DB."""
    try:
        ensure_context_tables(db)
        if not db["context_snapshots"].exists():
            return []
        rows: list[dict[str, Any]] = []
        for row in db["context_snapshots"].rows_where(
            where="run_id = ?",
            where_args=[run_id],
            order_by="step",
        ):
            breakdown: Any = []
            suggestions: Any = []
            try:
                breakdown = json.loads(row.get("breakdown_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                suggestions = json.loads(row.get("suggestions_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                pass
            rows.append(
                {
                    "step": row.get("step") or 0,
                    "model": row.get("model") or "unknown",
                    "context_limit": row.get("context_limit") or _DEFAULT_CONTEXT_LIMIT,
                    "tokens_used": row.get("tokens_used") or 0,
                    "pct_used": row.get("pct_used") or 0.0,
                    "alert_level": row.get("alert_level") or "ok",
                    "breakdown": breakdown,
                    "suggestions": suggestions,
                    "truncation_suspected": bool(row.get("truncation_suspected")),
                }
            )
        return rows
    except Exception:  # noqa: BLE001
        return []


def load_context_ui_data(db: Any) -> dict[str, Any]:
    """Aggregate context data for all runs — used by the UI."""
    result: dict[str, Any] = {"by_run": {}}
    if not db["runs"].exists():
        return result
    for run in db["runs"].rows_where(order_by="start_time desc", limit=100):
        rid = run["id"]
        snaps = load_context_snapshots(db, rid)
        if not snaps:
            continue
        latest = snaps[-1]
        result["by_run"][rid] = {
            "agent_name": run.get("agent_name") or "agent",
            "status": run.get("status") or "",
            "start_time": run.get("start_time") or "",
            "steps": snaps,
            "latest_pct": latest["pct_used"],
            "alert_level": latest["alert_level"],
            "model": latest["model"],
            "context_limit": latest["context_limit"],
        }
    return result


if __name__ == "__main__":
    db = get_db()
    create_tables(db)
    run_id = insert_run(db, agent_name="demo")
    mon = ContextMonitor(db=db, run_id=run_id)
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. " * 200},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    }
    result = mon.record_llm_call(payload, model="gpt-4o")
    print(result)
    print("Snapshots:", mon.get_usage())
