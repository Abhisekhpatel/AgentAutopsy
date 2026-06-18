"""Context Decay Detector for AgentAutopsy.

Tracks how much of the original session context survives into each subsequent
LLM step using three zero-config, embedding-free signals:

  1. Token overlap    — fraction of meaningful tokens from the first prompt
                        that still appear in the current prompt.
  2. Entity retention — key entities (quoted strings, proper names, all-caps IDs,
                        numbers, dates) extracted from the anchor step and checked
                        against later steps.
  3. Contradiction    — simple subject-verb negation patterns that contradict
                        claims made in the anchor step.

A health score 0–100 is computed per step and persisted to the
``context_health`` SQLite table.  A real-time warning is emitted when the
score drops below the configurable threshold (default 60).

Integration
-----------
``ContextHealthTracker.watch()`` uses the same monkey-patch technique as
``interceptor.py``: it wraps the already-patched ``Completions.create`` and
the Anthropic ``messages.create`` so that every LLM call in the session
automatically feeds into the health computation.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

# ── Stop-words filtered from token overlap (common words add noise) ───────────

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall", "not",
    "that", "this", "these", "those", "it", "its", "i", "you", "he", "she",
    "we", "they", "what", "which", "who", "how", "when", "where", "why",
    "if", "then", "than", "so", "no", "yes", "up", "out", "into", "just",
    "about", "also", "there", "here", "more", "all", "any", "each", "very",
    "said", "say", "says", "get", "got", "go", "going", "make", "use",
})

_WARN_THRESHOLD_DEFAULT = 60

# Module-level context — holds the currently registered tracker
_health_ctx: dict[str, Any] = {}

# ── Tokenisation ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Return the set of meaningful word tokens, excluding stop-words.

    Two token shapes are kept:
    - Alphabetic/mixed tokens starting with a letter (≥ 2 chars total).
    - Pure numeric tokens with ≥ 3 digits (e.g. "12345", "2024").

    Stop-words are filtered from the alphabetic set only; numbers are always
    kept because they carry high semantic signal (IDs, order numbers, dates).
    """
    alpha_tokens = re.findall(r"\b[a-z][a-z0-9_]{1,}\b", text.lower())
    numeric_tokens = re.findall(r"\b[0-9]{3,}\b", text)
    return {t for t in alpha_tokens if t not in _STOP_WORDS} | set(numeric_tokens)


# ── Entity extraction ─────────────────────────────────────────────────────────

_ENTITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'"[^"]{2,60}"'),                          # double-quoted strings
    re.compile(r"'[^']{2,60}'"),                          # single-quoted strings
    re.compile(r"\b[A-Z][A-Z0-9_\-]{2,}\b"),             # ALL-CAPS IDs / constants
    re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)+\b"),  # multi-word proper nouns
    re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"),          # ISO dates
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),    # common date formats
    re.compile(r"\b\d{6,}\b"),                            # long numeric IDs
    re.compile(r"\b[a-z][a-z0-9]*_[a-z][a-z0-9_]+\b"),  # snake_case identifiers
]


def _extract_entities(text: str) -> list[str]:
    """Return a deduplicated list of key entities from *text*."""
    found: list[str] = []
    seen: set[str] = set()
    for pat in _ENTITY_PATTERNS:
        for m in pat.finditer(text):
            val = m.group().strip("\"' ")
            key = val.lower()
            if key not in seen and len(key) >= 2:
                seen.add(key)
                found.append(val)
    return found


# ── Token overlap ─────────────────────────────────────────────────────────────

def _compute_token_overlap(anchor_tokens: set[str], current_tokens: set[str]) -> float:
    """Return fraction (0.0–1.0) of anchor tokens that appear in current tokens."""
    if not anchor_tokens:
        return 1.0
    return len(anchor_tokens & current_tokens) / len(anchor_tokens)


# ── Entity retention ──────────────────────────────────────────────────────────

def _compute_entity_retention(
    anchor_entities: list[str],
    current_text: str,
) -> tuple[float, list[str]]:
    """Return ``(retention_fraction, missing_entities)``."""
    if not anchor_entities:
        return 1.0, []
    current_lower = current_text.lower()
    missing: list[str] = []
    present = 0
    for entity in anchor_entities:
        if entity.lower() in current_lower:
            present += 1
        else:
            missing.append(entity)
    return present / len(anchor_entities), missing


# ── Contradiction detection ───────────────────────────────────────────────────

_NEGATION_FORMS = (
    " is not ", " isn't ", " are not ", " aren't ",
    " was not ", " wasn't ", " were not ", " weren't ",
    " has no ", " have no ", " cannot ", " can't ",
    " does not ", " doesn't ", " do not ", " don't ",
    " will not ", " won't ",
)


def _count_contradictions(anchor_text: str, current_text: str) -> int:
    """Count simple negation-based contradictions between anchor and current text.

    Extracts ``(subject, verb)`` bigrams from the anchor and checks whether
    the current text contains a negated form of the same subject-verb pair.
    Keeps a low false-positive rate by requiring at least a two-word match.
    """
    anchor_lower = anchor_text.lower()
    current_lower = current_text.lower()

    # Collect subject-verb pairs from anchor ("X is", "X has", "X can", ...)
    _SIMPLE_VERBS = {"is", "are", "has", "have", "can", "will", "does", "do", "was", "were"}
    words = re.findall(r"\b[a-z][a-z0-9_]{1,}\b", anchor_lower)
    anchor_pairs: set[tuple[str, str]] = set()
    for i in range(len(words) - 1):
        if words[i + 1] in _SIMPLE_VERBS and words[i] not in _STOP_WORDS:
            anchor_pairs.add((words[i], words[i + 1]))

    if not anchor_pairs:
        return 0

    count = 0
    for subj, verb in anchor_pairs:
        if f"{subj} {verb}" not in anchor_lower:
            continue
        # Check whether the negated form appears in the current text
        for negation in _NEGATION_FORMS:
            if f"{subj}{negation}" in current_lower or f" not {subj} " in current_lower:
                count += 1
                break
    return count


# ── Health scoring ────────────────────────────────────────────────────────────

def compute_health_score(
    token_overlap: float,
    entity_retention: float,
    contradiction_count: int,
) -> float:
    """Compute a weighted health score in the range 0–100.

    Weights: 50 % token overlap + 50 % entity retention, minus a capped
    contradiction penalty of up to 30 points.
    """
    raw = (0.5 * token_overlap + 0.5 * entity_retention) * 100.0
    penalty = min(30.0, contradiction_count * 10.0)
    return max(0.0, min(100.0, raw - penalty))


# ── DB schema ─────────────────────────────────────────────────────────────────

def ensure_context_health_tables(db: Any) -> None:
    """Create the ``context_health`` table if it does not already exist."""
    db["context_health"].create(
        {
            "id": str,
            "run_id": str,
            "step": int,
            "recorded_at": str,
            "health_score": float,
            "token_overlap": float,
            "entity_retention": float,
            "contradiction_count": int,
            "anchor_entity_count": int,
            "missing_entities": str,   # JSON list of entity strings
            "alert_fired": int,        # 1 when health < threshold
        },
        pk="id",
        if_not_exists=True,
    )


# ── Message flattening ────────────────────────────────────────────────────────

def _messages_to_text(messages: Any) -> str:
    """Flatten a messages list (or plain string) into one text blob."""
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return str(messages or "")
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, str):
            parts.append(msg)
        elif isinstance(msg, dict):
            content = msg.get("content") or ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        parts.append(str(block.get("text") or ""))
                    else:
                        parts.append(str(block))
            else:
                parts.append(str(content))
    return "\n".join(parts)


# ── ContextHealthTracker ──────────────────────────────────────────────────────

class ContextHealthTracker:
    """Track context health (decay) across LLM steps in a session.

    The tracker analyses every LLM call in the run by comparing its messages
    against the *anchor* (the very first call).  Three lightweight signals
    are combined into a 0–100 health score; a warning is printed when it
    falls below *warn_threshold*.

    Usage (automatic via ``agentautopsy.watch()``)::

        import agentautopsy
        agentautopsy.watch()          # registers ContextHealthTracker automatically

    Usage (manual)::

        tracker = ContextHealthTracker(db=db, run_id=run_id)
        tracker.watch()               # hooks interceptors
        # ... agent runs ...
        scores = tracker.get_scores() # [{step, health_score, ...}, ...]

    Parameters
    ----------
    db:
        sqlite-utils Database.  A new one is opened from cwd if ``None``.
    run_id:
        The current run ID.
    agent_name:
        Label used in console output.
    warn_threshold:
        Print a warning when health drops below this value (default 60).
    """

    def __init__(
        self,
        db: Any | None = None,
        run_id: str | None = None,
        *,
        agent_name: str = "agent",
        warn_threshold: int = _WARN_THRESHOLD_DEFAULT,
    ) -> None:
        self.db = db
        self.run_id = run_id
        self.agent_name = agent_name
        self.warn_threshold = warn_threshold

        self._lock = threading.Lock()
        self._step = 0
        self._anchor_tokens: set[str] = set()
        self._anchor_entities: list[str] = []
        self._anchor_text: str = ""
        self._scores: list[dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def watch(self) -> "ContextHealthTracker":
        """Register as the active tracker and hook into LLM interceptors.

        Wraps the already-patched ``Completions.create`` (from
        ``start_interceptor``) and Anthropic ``messages.create`` so that
        every subsequent LLM call feeds into :meth:`record_llm_call`
        automatically.
        """
        from agentautopsy.db import create_tables, get_db

        if self.db is None:
            self.db = get_db()
            create_tables(self.db)
        ensure_context_health_tables(self.db)
        _health_ctx["tracker"] = self
        self._patch_openai()
        self._patch_anthropic()
        return self

    def record_llm_call(
        self,
        messages: Any,
        *,
        model: str = "unknown",
    ) -> dict[str, Any]:
        """Analyse one LLM call and update the running health score.

        Step 1 is always treated as the anchor (health = 100).  Subsequent
        steps are scored relative to it.

        Returns a snap dict with ``health_score``, ``missing_entities``, etc.
        """
        with self._lock:
            self._step += 1
            step = self._step

        text = _messages_to_text(messages)

        if step == 1:
            # Anchor step: establish baseline
            with self._lock:
                self._anchor_text = text
                self._anchor_tokens = _tokenize(text)
                self._anchor_entities = _extract_entities(text)
            token_ov = 1.0
            entity_ret = 1.0
            contradictions = 0
            missing: list[str] = []
            score_val = 100.0
        else:
            with self._lock:
                anchor_tokens = self._anchor_tokens
                anchor_entities = self._anchor_entities
                anchor_text = self._anchor_text

            current_tokens = _tokenize(text)
            token_ov = _compute_token_overlap(anchor_tokens, current_tokens)
            entity_ret, missing = _compute_entity_retention(anchor_entities, text)
            contradictions = _count_contradictions(anchor_text, text)
            score_val = compute_health_score(token_ov, entity_ret, contradictions)

        snap: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "run_id": self.run_id or "",
            "step": step,
            "health_score": round(score_val, 2),
            "token_overlap": round(token_ov * 100, 2),
            "entity_retention": round(entity_ret * 100, 2),
            "contradiction_count": contradictions,
            "anchor_entity_count": len(self._anchor_entities),
            "missing_entities": missing,
            "alert_fired": 0,
        }

        # Warn if health has dropped below threshold (never warn on anchor step)
        if step > 1 and score_val < self.warn_threshold:
            snap["alert_fired"] = 1
            self._print_warning(snap)

        with self._lock:
            self._scores.append(snap)

        self._persist(snap)
        return snap

    def get_scores(self) -> list[dict[str, Any]]:
        """Return all per-step health records accumulated so far."""
        with self._lock:
            return list(self._scores)

    def current_score(self) -> float:
        """Latest health score (0–100), or 100.0 if no steps recorded yet."""
        with self._lock:
            if not self._scores:
                return 100.0
            return self._scores[-1]["health_score"]

    def reset(self) -> None:
        """Clear all state so the tracker can be reused for a new run."""
        with self._lock:
            self._step = 0
            self._anchor_tokens = set()
            self._anchor_entities = []
            self._anchor_text = ""
            self._scores.clear()

    # ── Interceptor patching ──────────────────────────────────────────────────

    def _patch_openai(self) -> None:
        """Wrap the existing OpenAI Completions.create with a health hook.

        Uses a guard flag so the patch is applied at most once per process.
        """
        try:
            from openai.resources.chat.completions import Completions
        except ImportError:
            return

        if getattr(Completions, "_agentautopsy_health_patched", False):
            return

        _upstream = Completions.create

        def _health_create(*args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages") or []
            model = str(kwargs.get("model") or "unknown")
            response = _upstream(*args, **kwargs)
            active = get_active_tracker()
            if active is not None:
                try:
                    active.record_llm_call(messages, model=model)
                except Exception:  # noqa: BLE001
                    pass
            return response

        Completions.create = _health_create
        Completions._agentautopsy_health_patched = True  # type: ignore[attr-defined]

    def _patch_anthropic(self) -> None:
        """Wrap the existing Anthropic __init__ to add a health hook around
        ``messages.create``.
        """
        try:
            import anthropic
        except ImportError:
            return

        client_class = anthropic.Anthropic
        if getattr(client_class, "_agentautopsy_health_patched", False):
            return

        original_init = client_class.__init__

        def _health_init(self_client: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self_client, *args, **kwargs)
            _upstream_create = self_client.messages.create

            def _health_messages_create(*a: Any, **kw: Any) -> Any:
                messages = kw.get("messages") or []
                model = str(kw.get("model") or "unknown")
                response = _upstream_create(*a, **kw)
                active = get_active_tracker()
                if active is not None:
                    try:
                        active.record_llm_call(messages, model=model)
                    except Exception:  # noqa: BLE001
                        pass
                return response

            self_client.messages.create = _health_messages_create

        client_class.__init__ = _health_init
        client_class._agentautopsy_health_patched = True  # type: ignore[attr-defined]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _persist(self, snap: dict[str, Any]) -> None:
        if self.db is None or not self.run_id:
            return
        try:
            ensure_context_health_tables(self.db)
            self.db["context_health"].insert(
                {
                    "id": snap["id"],
                    "run_id": snap["run_id"],
                    "step": snap["step"],
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "health_score": snap["health_score"],
                    "token_overlap": snap["token_overlap"],
                    "entity_retention": snap["entity_retention"],
                    "contradiction_count": snap["contradiction_count"],
                    "anchor_entity_count": snap["anchor_entity_count"],
                    "missing_entities": json.dumps(snap["missing_entities"]),
                    "alert_fired": snap["alert_fired"],
                },
                pk="id",
            )
        except Exception:  # noqa: BLE001
            pass

    def _print_warning(self, snap: dict[str, Any]) -> None:
        step = snap["step"]
        total = len(self._scores) + 1   # _scores hasn't been updated yet for this step
        score = int(snap["health_score"])
        missing = snap["missing_entities"]

        entity_suffix = ""
        if missing:
            quoted = ", ".join(f"'{e}'" for e in missing[:3])
            entity_suffix = f" Key entity {quoted} lost."

        print(
            f"\n\033[38;5;208m[AgentAutopsy] WARNING — context health at {score}%"
            f" (step {step} of {total}).{entity_suffix}\033[0m"
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def get_active_tracker() -> ContextHealthTracker | None:
    """Return the currently registered ContextHealthTracker, or None."""
    tracker = _health_ctx.get("tracker")
    return tracker if isinstance(tracker, ContextHealthTracker) else None


def record_health_event(
    messages: Any,
    *,
    model: str = "unknown",
) -> dict[str, Any]:
    """Feed an LLM call to the active ContextHealthTracker if one is registered."""
    tracker = get_active_tracker()
    if tracker is None:
        return {"ok": True}
    return tracker.record_llm_call(messages, model=model)


def load_health_scores(db: Any, run_id: str) -> list[dict[str, Any]]:
    """Load persisted health scores for *run_id*, ordered by step number."""
    try:
        ensure_context_health_tables(db)
        if not db["context_health"].exists():
            return []
        rows: list[dict[str, Any]] = []
        for row in db["context_health"].rows_where(
            where="run_id = ?",
            where_args=[run_id],
            order_by="step",
        ):
            missing: list[str] = []
            try:
                missing = json.loads(row.get("missing_entities") or "[]")
            except (json.JSONDecodeError, TypeError):
                pass
            rows.append(
                {
                    "step": row.get("step") or 0,
                    "health_score": row.get("health_score") or 100.0,
                    "token_overlap": row.get("token_overlap") or 100.0,
                    "entity_retention": row.get("entity_retention") or 100.0,
                    "contradiction_count": row.get("contradiction_count") or 0,
                    "anchor_entity_count": row.get("anchor_entity_count") or 0,
                    "missing_entities": missing,
                    "alert_fired": bool(row.get("alert_fired")),
                    "recorded_at": row.get("recorded_at") or "",
                }
            )
        return rows
    except Exception:  # noqa: BLE001
        return []
