"""Runaway loop detector and cost kill switch for AgentAutopsy.

Monitors every tool call and LLM call in real time. Kills the agent and
saves the trace the moment a runaway loop or cost threshold is crossed.
Once AgentAutopsy has stopped your first $400 bill you can never remove it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from agentautopsy.db import create_tables, get_db, insert_event, insert_run

# ── Model pricing table (USD per 1M tokens) ──────────────────────────────────
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    "gpt-4o":                (5.00,  15.00),
    "gpt-4o-mini":           (0.15,   0.60),
    "claude-opus-4-6":       (15.00,  75.00),
    "claude-sonnet-4-6":     (3.00,  15.00),
    "claude-haiku-4-5":      (0.80,   4.00),
    "gemini-1.5-pro":        (3.50,  10.50),
    # aliases / common shorthands
    "gpt-4":                 (30.00,  60.00),
    "gpt-3.5-turbo":         (0.50,   1.50),
    "claude-3-opus":         (15.00,  75.00),
    "claude-3-sonnet":       (3.00,  15.00),
    "claude-3-haiku":        (0.25,   1.25),
    "gemini-1.5-flash":      (0.075,  0.30),
}

# Default pricing applied when the model is not in the table
_DEFAULT_PRICING: tuple[float, float] = (5.00, 15.00)

# Global context — one LoopDetector may be active per process
_loop_context: dict[str, Any] = {}

# Allowed call types that count as "a step"
TRACKABLE_CALL_TYPES: frozenset[str] = frozenset(
    {
        "llm_call",
        "llm_response",
        "tool_call",
        "tool_result",
        "mcp_tool_call",
        "mcp_tool_result",
        "http_request",
        "http_response",
    }
)

REPEAT_CALL_TYPES: frozenset[str] = frozenset({"tool_call", "mcp_tool_call"})
LLM_CALL_TYPES: frozenset[str] = frozenset({"llm_call", "llm_response"})


def _cost_usd(
    model: str,
    token_input: int,
    token_output: int,
) -> float:
    key = model.lower().strip()
    # Try exact match first, then prefix match
    pricing = MODEL_PRICING.get(key)
    if pricing is None:
        for prefix, p in MODEL_PRICING.items():
            if key.startswith(prefix) or prefix.startswith(key):
                pricing = p
                break
    if pricing is None:
        pricing = _DEFAULT_PRICING
    per_input, per_output = pricing
    return (token_input * per_input + token_output * per_output) / 1_000_000.0


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    """Stable hash of the semantically-meaningful parts of a call payload."""
    relevant = {
        k: payload[k]
        for k in ("messages", "input", "args", "arguments", "query", "prompt")
        if k in payload and payload[k] not in (None, "", [], {})
    }
    raw = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _tool_name(payload: dict[str, Any]) -> str:
    for k in ("tool", "name", "tool_name", "function"):
        if k in payload and payload[k]:
            return str(payload[k])
    return "unknown_tool"


def ensure_loop_tables(db: Any) -> None:
    """Create the loop_events table if it does not exist."""
    db["loop_events"].create(
        {
            "id": str,
            "run_id": str,
            "detected_at": str,
            "loop_type": str,
            "trigger_step": int,
            "trigger_label": str,
            "total_steps": int,
            "total_tokens": int,
            "total_cost_usd": float,
            "detail_json": str,
            "killed": int,
        },
        pk="id",
        if_not_exists=True,
    )


class LoopKillException(RuntimeError):
    """Raised when LoopDetector hard-stops the agent."""

    def __init__(self, reason: str, loop_type: str) -> None:
        super().__init__(reason)
        self.loop_type = loop_type


class LoopDetector:
    """Real-time runaway loop detector and cost kill switch.

    Tracks every tool call and LLM call recorded by AgentAutopsy. When a
    threshold is crossed it inserts a ``loop_event`` row, optionally raises
    ``LoopKillException``, and auto-generates an eval test.

    Parameters
    ----------
    max_iterations:
        Hard stop after this many total tracked steps (default 20).
    max_cost_usd:
        Kill the agent when cumulative cost reaches this value (default $1.00).
    max_tokens:
        Kill the agent when total tokens reach this value (default 100 000).
    max_repeat_calls:
        Flag a loop when the *same tool* is called this many consecutive times
        (default 3).
    max_repeat_llm:
        Flag a stuck loop when the *same LLM input fingerprint* repeats this
        many times (default 2).
    max_recursion:
        Flag infinite recursion when the same agent calls itself this many
        times (default 3).
    warn_at_fraction:
        Issue a warning when cost/tokens reach this fraction of the threshold
        (default 0.8 = 80 %).
    kill_on_loop:
        Raise ``LoopKillException`` when a loop is detected (default True).
    """

    def __init__(
        self,
        db: Any | None = None,
        run_id: str | None = None,
        *,
        agent_name: str = "agent",
        max_iterations: int = 20,
        max_cost_usd: float = 1.00,
        max_tokens: int = 100_000,
        max_repeat_calls: int = 3,
        max_repeat_llm: int = 2,
        max_recursion: int = 3,
        warn_at_fraction: float = 0.80,
        kill_on_loop: bool = True,
    ) -> None:
        self.db = db
        self.run_id = run_id
        self.agent_name = agent_name
        self.max_iterations = max_iterations
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens
        self.max_repeat_calls = max_repeat_calls
        self.max_repeat_llm = max_repeat_llm
        self.max_recursion = max_recursion
        self.warn_at_fraction = warn_at_fraction
        self.kill_on_loop = kill_on_loop

        # Runtime counters — thread-safe via a lock
        self._lock = threading.Lock()
        self._steps: int = 0
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._consecutive_tool: list[str] = []  # recent tool names
        self._llm_fingerprints: list[str] = []  # recent LLM input hashes
        self._recursion_counts: dict[str, int] = {}  # agent_name → depth
        self._loop_events: list[dict[str, Any]] = []
        self._warned_cost: bool = False
        self._warned_tokens: bool = False
        self._killed: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def watch(self) -> LoopDetector:
        """Register as the active detector and start monitoring the DB."""
        if self.db is None:
            self.db = get_db()
        create_tables(self.db)
        ensure_loop_tables(self.db)
        if self.run_id is None:
            self.run_id = insert_run(self.db, agent_name=self.agent_name)
        _loop_context["detector"] = self
        return self

    def record_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        token_input: int = 0,
        token_output: int = 0,
        model: str = "gpt-4o",
        step_number: int | None = None,
        calling_agent: str | None = None,
    ) -> dict[str, Any]:
        """Feed a single event into the detector.

        Returns a status dict:
        ``{"ok": True}`` when no issue is found, or
        ``{"ok": False, "loop_type": ..., "reason": ..., "warning": bool}``
        when a threshold is crossed.
        """
        if self._killed:
            return {"ok": False, "loop_type": "already_killed", "reason": "Agent already killed", "warning": False}

        with self._lock:
            # ── Step counting ─────────────────────────────────────────────
            if event_type in TRACKABLE_CALL_TYPES:
                self._steps += 1

            # ── Token + cost accounting ───────────────────────────────────
            self._total_tokens += token_input + token_output
            if token_input or token_output:
                self._total_cost_usd += _cost_usd(model, token_input, token_output)

            # ── Tool-call repeat tracking ─────────────────────────────────
            if event_type in REPEAT_CALL_TYPES:
                name = _tool_name(payload)
                self._consecutive_tool.append(name)
                # Keep only a window of (max_repeat_calls + 1) entries
                self._consecutive_tool = self._consecutive_tool[-(self.max_repeat_calls + 1):]
            elif event_type in LLM_CALL_TYPES:
                # Non-tool event resets consecutive tool streak
                self._consecutive_tool.clear()

            # ── LLM fingerprint tracking ──────────────────────────────────
            if event_type in LLM_CALL_TYPES:
                fp = _payload_fingerprint(payload)
                self._llm_fingerprints.append(fp)
                self._llm_fingerprints = self._llm_fingerprints[-(self.max_repeat_llm + 1):]

            # ── Recursion tracking ────────────────────────────────────────
            agent = calling_agent or self.agent_name
            if event_type == "agent_call" and calling_agent:
                self._recursion_counts[agent] = self._recursion_counts.get(agent, 0) + 1

            step = step_number or self._steps
            result = self._evaluate(step, payload)
            return result

    def current_stats(self) -> dict[str, Any]:
        """Return current counters (thread-safe snapshot)."""
        with self._lock:
            return {
                "steps": self._steps,
                "total_tokens": self._total_tokens,
                "total_cost_usd": round(self._total_cost_usd, 6),
                "cost_pct": min(100, round(self._total_cost_usd / self.max_cost_usd * 100, 1)),
                "token_pct": min(100, round(self._total_tokens / self.max_tokens * 100, 1)),
                "loop_events": list(self._loop_events),
                "killed": self._killed,
            }

    def reset(self) -> None:
        """Reset all counters for a new run."""
        with self._lock:
            self._steps = 0
            self._total_tokens = 0
            self._total_cost_usd = 0.0
            self._consecutive_tool.clear()
            self._llm_fingerprints.clear()
            self._recursion_counts.clear()
            self._loop_events.clear()
            self._warned_cost = False
            self._warned_tokens = False
            self._killed = False

    # ── Internal evaluation ───────────────────────────────────────────────────

    def _evaluate(
        self, step: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Run all checks; return first triggered issue or ok."""

        # 1. Repeated tool calls
        if (
            len(self._consecutive_tool) >= self.max_repeat_calls
            and len(set(self._consecutive_tool[-self.max_repeat_calls:])) == 1
        ):
            tool = self._consecutive_tool[-1]
            return self._trigger(
                loop_type="repeated_tool_call",
                step=step,
                label=f"{tool}() called {self.max_repeat_calls}× in a row",
                detail={"tool": tool, "count": self.max_repeat_calls},
                kill=True,
            )

        # 2. Stuck LLM loop — same input repeated
        if (
            len(self._llm_fingerprints) >= self.max_repeat_llm
            and len(set(self._llm_fingerprints[-self.max_repeat_llm:])) == 1
        ):
            fp = self._llm_fingerprints[-1]
            return self._trigger(
                loop_type="stuck_llm_loop",
                step=step,
                label=f"LLM received the same input {self.max_repeat_llm}× (fingerprint {fp})",
                detail={"fingerprint": fp, "count": self.max_repeat_llm},
                kill=True,
            )

        # 3. Max iterations hard stop
        if self._steps > self.max_iterations:
            return self._trigger(
                loop_type="max_iterations",
                step=step,
                label=f"Step {step} exceeded max_iterations={self.max_iterations}",
                detail={"steps": self._steps, "limit": self.max_iterations},
                kill=True,
            )

        # 4. Infinite recursion
        for agent_name, depth in self._recursion_counts.items():
            if depth > self.max_recursion:
                return self._trigger(
                    loop_type="infinite_recursion",
                    step=step,
                    label=f"Agent '{agent_name}' called itself {depth} times",
                    detail={"agent": agent_name, "depth": depth},
                    kill=True,
                )

        # 5. Cost kill switch
        cost_warn_threshold = self.max_cost_usd * self.warn_at_fraction
        if self._total_cost_usd >= self.max_cost_usd:
            return self._trigger(
                loop_type="cost_exceeded",
                step=step,
                label=f"Cost ${self._total_cost_usd:.4f} exceeded limit ${self.max_cost_usd:.2f}",
                detail={"cost_usd": self._total_cost_usd, "limit_usd": self.max_cost_usd},
                kill=True,
            )
        if not self._warned_cost and self._total_cost_usd >= cost_warn_threshold:
            self._warned_cost = True
            return self._warn(
                loop_type="cost_warning",
                step=step,
                label=f"Cost ${self._total_cost_usd:.4f} reached "
                      f"{int(self.warn_at_fraction * 100)}% of ${self.max_cost_usd:.2f} limit",
                detail={"cost_usd": self._total_cost_usd, "limit_usd": self.max_cost_usd,
                        "pct": int(self.warn_at_fraction * 100)},
            )

        # 6. Token kill switch
        token_warn_threshold = int(self.max_tokens * self.warn_at_fraction)
        if self._total_tokens >= self.max_tokens:
            return self._trigger(
                loop_type="tokens_exceeded",
                step=step,
                label=f"Token count {self._total_tokens} exceeded limit {self.max_tokens}",
                detail={"tokens": self._total_tokens, "limit": self.max_tokens},
                kill=True,
            )
        if not self._warned_tokens and self._total_tokens >= token_warn_threshold:
            self._warned_tokens = True
            return self._warn(
                loop_type="token_warning",
                step=step,
                label=f"Tokens {self._total_tokens} reached "
                      f"{int(self.warn_at_fraction * 100)}% of {self.max_tokens} limit",
                detail={"tokens": self._total_tokens, "limit": self.max_tokens,
                        "pct": int(self.warn_at_fraction * 100)},
            )

        return {"ok": True}

    def _warn(
        self, loop_type: str, step: int, label: str, detail: dict[str, Any]
    ) -> dict[str, Any]:
        """Record a non-fatal warning in the event stream and return it."""
        self._record_loop_event(loop_type, step, label, detail, killed=False)
        self._print_warning(loop_type, label)
        return {"ok": False, "loop_type": loop_type, "reason": label, "warning": True}

    def _trigger(
        self, loop_type: str, step: int, label: str, detail: dict[str, Any], kill: bool
    ) -> dict[str, Any]:
        """Record a loop event, kill the agent if configured, return status."""
        self._record_loop_event(loop_type, step, label, detail, killed=kill and self.kill_on_loop)
        self._killed = kill and self.kill_on_loop
        self._print_alert(loop_type, label)
        self._generate_eval_if_possible()
        if self._killed:
            raise LoopKillException(label, loop_type)
        return {"ok": False, "loop_type": loop_type, "reason": label, "warning": False}

    def _record_loop_event(
        self,
        loop_type: str,
        step: int,
        label: str,
        detail: dict[str, Any],
        *,
        killed: bool,
    ) -> None:
        import uuid

        now = datetime.now(timezone.utc).isoformat()
        event_id = str(uuid.uuid4())
        entry: dict[str, Any] = {
            "id": event_id,
            "loop_type": loop_type,
            "trigger_step": step,
            "trigger_label": label,
            "total_steps": self._steps,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost_usd, 6),
            "detail": detail,
        }
        self._loop_events.append(entry)

        if self.db is not None:
            try:
                ensure_loop_tables(self.db)
                self.db["loop_events"].insert(
                    {
                        "id": event_id,
                        "run_id": self.run_id or "",
                        "detected_at": now,
                        "loop_type": loop_type,
                        "trigger_step": step,
                        "trigger_label": label,
                        "total_steps": self._steps,
                        "total_tokens": self._total_tokens,
                        "total_cost_usd": round(self._total_cost_usd, 6),
                        "detail_json": json.dumps(detail),
                        "killed": int(killed),
                    },
                    pk="id",
                )
                if self.run_id:
                    insert_event(
                        self.db,
                        self.run_id,
                        "loop_detected",
                        {
                            "loop_type": loop_type,
                            "trigger_step": step,
                            "label": label,
                            "killed": killed,
                            **detail,
                        },
                    )
            except Exception:  # noqa: BLE001
                pass

    def _generate_eval_if_possible(self) -> None:
        if self.run_id is None or self.db is None:
            return
        try:
            from agentautopsy.db import mark_run_failed
            from agentautopsy.eval_generator import generate_eval_for_run

            mark_run_failed(self.db, self.run_id)
            path = generate_eval_for_run(self.run_id, self.db)
            if path:
                print(
                    f"\033[38;5;244m▶ Eval:  \033[1;38;5;82mLoop regression test → {path}\033[0m"
                )
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _print_warning(loop_type: str, label: str) -> None:
        print(
            f"\n\033[38;5;226m⚠ [AgentAutopsy] Loop Warning [{loop_type}]\033[0m"
        )
        print(f"\033[38;5;244m▶ {label}\033[0m")

    @staticmethod
    def _print_alert(loop_type: str, label: str) -> None:
        print(
            f"\n\033[1;38;5;196m🛑 [AgentAutopsy] Loop Killed [{loop_type}]\033[0m"
        )
        print(f"\033[38;5;196m▶ {label}\033[0m")


# ── Module-level helpers ──────────────────────────────────────────────────────

def get_active_detector() -> LoopDetector | None:
    detector = _loop_context.get("detector")
    return detector if isinstance(detector, LoopDetector) else None


def record_call_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    token_input: int = 0,
    token_output: int = 0,
    model: str = "gpt-4o",
    step_number: int | None = None,
    calling_agent: str | None = None,
) -> dict[str, Any]:
    """Feed an event to the active LoopDetector if one is registered."""
    detector = get_active_detector()
    if detector is None:
        return {"ok": True}
    return detector.record_event(
        event_type,
        payload,
        token_input=token_input,
        token_output=token_output,
        model=model,
        step_number=step_number,
        calling_agent=calling_agent,
    )


def load_loop_events(db: Any) -> list[dict[str, Any]]:
    """Load all loop events from the database for the UI / CLI."""
    try:
        ensure_loop_tables(db)
        if not db["loop_events"].exists():
            return []
        rows: list[dict[str, Any]] = []
        for row in db["loop_events"].rows_where(order_by="detected_at desc"):
            detail: Any = {}
            try:
                detail = json.loads(row.get("detail_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
            rows.append(
                {
                    "id": row["id"],
                    "run_id": row.get("run_id") or "",
                    "detected_at": row.get("detected_at") or "",
                    "loop_type": row.get("loop_type") or "",
                    "trigger_step": row.get("trigger_step") or 0,
                    "trigger_label": row.get("trigger_label") or "",
                    "total_steps": row.get("total_steps") or 0,
                    "total_tokens": row.get("total_tokens") or 0,
                    "total_cost_usd": row.get("total_cost_usd") or 0.0,
                    "killed": bool(row.get("killed")),
                    "detail": detail,
                }
            )
        return rows
    except Exception:  # noqa: BLE001
        return []


def cost_per_run(db: Any) -> list[dict[str, Any]]:
    """Return per-run cost / token breakdown for the UI."""
    if not db["events"].exists() or not db["runs"].exists():
        return []
    result: list[dict[str, Any]] = []
    for run in db["runs"].rows_where(order_by="start_time desc"):
        run_id = run["id"]
        row = db.execute(
            "SELECT COALESCE(SUM(token_input),0), COALESCE(SUM(token_output),0), "
            "COALESCE(SUM(cost_usd),0.0) FROM events WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            continue
        ti, to, cost = int(row[0]), int(row[1]), float(row[2])
        result.append(
            {
                "run_id": run_id,
                "agent_name": run.get("agent_name") or "agent",
                "status": run.get("status") or "",
                "start_time": run.get("start_time") or "",
                "token_input": ti,
                "token_output": to,
                "total_tokens": ti + to,
                "cost_usd": round(cost, 6),
            }
        )
    return result


if __name__ == "__main__":
    db = get_db()
    create_tables(db)
    run_id = insert_run(db, agent_name="demo")
    det = LoopDetector(db=db, run_id=run_id, max_repeat_calls=3, kill_on_loop=False)
    det.watch()

    for i in range(4):
        status = det.record_event("tool_call", {"tool": "search"}, token_input=100, token_output=50)
        print(f"step {i+1}: {status}")

    print("stats:", det.current_stats())
