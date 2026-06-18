"""Failure DNA Fingerprinting for AgentAutopsy.

Generates a short, normalized fingerprint ID (e.g. ``FP-7a3f9c``) for each
distinct failure pattern.  Structurally identical failures — same error type,
same failing tool, same root-cause shape — always produce the same ID even when
volatile tokens (run IDs, timestamps, URLs, numeric values) differ between runs.

The ``failure_fingerprints`` table persists each pattern so that a repeat failure
can be identified instantly and its stored fix returned without re-running AI
analysis.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from sqlite_utils import Database

# ---------------------------------------------------------------------------
# Normalization: strip tokens that vary per run but not per pattern
# ---------------------------------------------------------------------------
_NORMALIZE_SUBS: list[tuple[re.Pattern[str], str]] = [
    # UUIDs  (e.g. run IDs, event IDs)
    (
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            re.IGNORECASE,
        ),
        "<id>",
    ),
    # ISO-8601 timestamps — use [Tt ] because input is always lowercased first
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}"
            r"(?:\.\d+)?(?:[Zz]|[+-]\d{2}:?\d{2})?\b"
        ),
        "<timestamp>",
    ),
    # URLs
    (re.compile(r"https?://\S+"), "<url>"),
    # IPv4 addresses (optionally with port)
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<ip>"),
    # Standalone integers (not inside words like "gpt-4" or "v2")
    (re.compile(r"(?<![a-z\-])\b\d+\b(?![a-z\-])"), "<N>"),
]

_ANSI_CYAN = "\033[96m"
_ANSI_GREEN = "\033[92m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"

# ---------------------------------------------------------------------------
# Patterns used only inside generate_fingerprint to strip whole key-value
# metadata fields before hashing.  This makes structurally identical errors
# (e.g. "timed out" vs "timed out run=<uuid> at <timestamp>") hash the same.
# ---------------------------------------------------------------------------

# Matches "key=UUID" including any preceding whitespace, e.g. " run=550e8400-..."
_KV_UUID = re.compile(
    r"\s*\b\w+\s*=\s*"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

# Matches "keyword TIMESTAMP" including any preceding whitespace,
# e.g. " at 2025-06-01t10:00:00z" (already lowercased at call site)
_KW_TIMESTAMP = re.compile(
    r"\s*\b(?:at|since|from|for|after|during|on)\s+"
    r"\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:[Zz]|[+-]\d{2}:?\d{2})?\b"
)


# ---------------------------------------------------------------------------
# Public: normalization
# ---------------------------------------------------------------------------

def normalize_root_cause(text: str) -> str:
    """Strip volatile tokens so structurally identical messages hash identically.

    Replaces UUIDs → ``<id>``, ISO timestamps → ``<timestamp>``, URLs →
    ``<url>``, IPs → ``<ip>``, and standalone numbers → ``<N>``.

    Examples
    --------
    >>> normalize_root_cause("request timed out after 30s (run=abc-123)")
    'request timed out after <N>s (run=<id>)'
    """
    text = text.lower().strip()
    for pattern, replacement in _NORMALIZE_SUBS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_hashing(text: str) -> str:
    """Normalise a failure message for use as the fingerprint hash seed.

    More aggressive than :func:`normalize_root_cause`: first strips whole
    ``key=UUID`` and ``keyword TIMESTAMP`` metadata phrases that are
    appended to the core error message (e.g. ``run=<uuid> at <timestamp>``),
    then applies the standard volatile-token replacements.

    This ensures that structurally identical errors produce the same hash
    even when different volatile metadata is appended:

    >>> _normalize_for_hashing("timed out after 30s")
    'timed out after 30s'
    >>> _normalize_for_hashing("timed out after 30s run=550e8400-... at 2025-06-01T10:00:00Z")
    'timed out after 30s'
    """
    text = text.lower().strip()
    # Remove "key=UUID" pairs (e.g. "run=550e8400-...") including leading whitespace
    text = _KV_UUID.sub("", text)
    # Remove "keyword TIMESTAMP" phrases (e.g. "at 2025-06-01t10:00:00z") including leading ws
    text = _KW_TIMESTAMP.sub("", text)
    # Apply the standard placeholder replacements for any remaining volatile tokens
    for pattern, replacement in _NORMALIZE_SUBS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _find_failing_tool(events: list[dict[str, Any]]) -> str:
    """Return the name of the last tool called before the first error event.

    Falls back to ``"unknown_tool"`` if no tool call precedes the error.
    """
    last_tool = "unknown_tool"
    for event in events:
        ev_type = event.get("type", "")
        if ev_type == "tool_call":
            payload = _parse_payload(event.get("payload"))
            name = (
                payload.get("tool")
                or payload.get("name")
                or payload.get("function")
                or "unknown_tool"
            )
            last_tool = str(name)
        elif ev_type in ("error", "http_error"):
            break
    return last_tool


def _failing_event_index(events: list[dict[str, Any]]) -> int:
    """Return the index of the first error/http_error event, or the last index."""
    for i, event in enumerate(events):
        if event.get("type") in ("error", "http_error"):
            return i
    return max(0, len(events) - 1)


def _step_position(failing_index: int, total: int) -> str:
    """Map a failing event position to ``'early'`` | ``'mid'`` | ``'late'``."""
    if total <= 0:
        return "early"
    ratio = failing_index / total
    if ratio < 0.33:
        return "early"
    if ratio < 0.67:
        return "mid"
    return "late"


def _confidence(occurrence_count: int, has_fix: bool) -> str:
    """Compute confidence from occurrence frequency and whether a fix is stored."""
    if occurrence_count >= 5 and has_fix:
        return "high"
    if occurrence_count >= 3 or (occurrence_count >= 2 and has_fix):
        return "medium"
    return "low"


def _unpack_trace(
    trace: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(failure_dict, events_list)`` from either trace layout.

    Accepts both:
    - ``{"failure": {...}, "events": [...]}``  (structured)
    - ``{"error_type": ..., "message": ..., "events": [...]}``  (flat)
    """
    if "failure" in trace and isinstance(trace["failure"], dict):
        return trace["failure"], trace.get("events") or []
    return trace, trace.get("events") or []


# ---------------------------------------------------------------------------
# Public: core fingerprinting API
# ---------------------------------------------------------------------------

def generate_fingerprint(trace: dict[str, Any]) -> str:
    """Create a normalized fingerprint ID for a failed run.

    The fingerprint is derived from four components that are stable across
    structurally identical failures:

    1. **Error type** — e.g. ``TimeoutError``, ``APIConnectionError``
    2. **Failing tool** — the last ``tool_call`` event before the first error
    3. **Normalized root cause** — the error message with volatile tokens stripped
    4. **Step position** — where in the run the failure occurred (``early`` /
       ``mid`` / ``late``)

    Parameters
    ----------
    trace:
        A dict with either a nested ``"failure"`` key or flat ``error_type`` /
        ``message`` keys, plus an ``"events"`` list of ``{"type", "payload"}``
        dicts (the same format used by ``analyzer.py``).

    Returns
    -------
    str
        A short fingerprint ID, e.g. ``"FP-7a3f9c"``.
    """
    failure, events = _unpack_trace(trace)

    error_type = str(failure.get("error_type") or "UnknownError")
    message = str(failure.get("message") or "")

    failing_tool = _find_failing_tool(events)
    normalized_cause = _normalize_for_hashing(message)
    fail_idx = _failing_event_index(events)
    position = _step_position(fail_idx, len(events))

    raw = f"{error_type}|{failing_tool}|{normalized_cause}|{position}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"FP-{digest[:6]}"


def ensure_fingerprint_tables(db: Database) -> None:
    """Create the ``failure_fingerprints`` table if it does not already exist."""
    db["failure_fingerprints"].create(
        {
            "fingerprint_id": str,      # "FP-7a3f9c"
            "error_type": str,          # "TimeoutError"
            "failing_tool": str,        # "get_weather"
            "step_position": str,       # "early" | "mid" | "late"
            "first_seen_at": str,       # ISO-8601
            "last_seen_at": str,        # ISO-8601
            "occurrence_count": int,    # total times seen
            "example_root_cause": str,  # raw error message from first occurrence
            "example_fix": str,         # fix text once a verified fix is known
            "confidence": str,          # "low" | "medium" | "high"
        },
        pk="fingerprint_id",
        if_not_exists=True,
    )


def match_fingerprint(db: Database, fp_id: str) -> dict[str, Any] | None:
    """Return the stored fingerprint record, or ``None`` if not seen before."""
    if not db["failure_fingerprints"].exists():
        return None
    try:
        row = db["failure_fingerprints"].get(fp_id)
        return dict(row) if row is not None else None
    except Exception:
        return None


def record_fingerprint(
    db: Database,
    fp_id: str,
    trace: dict[str, Any],
    *,
    fix: str | None = None,
) -> dict[str, Any]:
    """Insert a new fingerprint row or increment its ``occurrence_count``.

    On the first occurrence an empty row is created.  On subsequent occurrences
    ``occurrence_count``, ``last_seen_at``, and ``confidence`` are updated.
    ``example_fix`` is written on first occurrence if *fix* is provided, and
    is never overwritten once set (use :func:`update_fingerprint_fix` instead).

    Parameters
    ----------
    db:
        Active sqlite-utils Database.
    fp_id:
        Fingerprint ID from :func:`generate_fingerprint`.
    trace:
        The same trace dict passed to :func:`generate_fingerprint` so that
        ``failing_tool``, ``step_position``, and ``example_root_cause`` can be
        extracted.
    fix:
        Optional verified fix text to store on first occurrence.

    Returns
    -------
    dict
        The up-to-date fingerprint record.
    """
    ensure_fingerprint_tables(db)
    now = datetime.now(timezone.utc).isoformat()

    failure, events = _unpack_trace(trace)
    error_type = str(failure.get("error_type") or "UnknownError")
    root_cause = str(failure.get("message") or "")
    failing_tool = _find_failing_tool(events)
    fail_idx = _failing_event_index(events)
    position = _step_position(fail_idx, len(events))

    existing = match_fingerprint(db, fp_id)

    if existing is None:
        row: dict[str, Any] = {
            "fingerprint_id": fp_id,
            "error_type": error_type,
            "failing_tool": failing_tool,
            "step_position": position,
            "first_seen_at": now,
            "last_seen_at": now,
            "occurrence_count": 1,
            "example_root_cause": root_cause,
            "example_fix": fix or "",
            "confidence": _confidence(1, bool(fix)),
        }
        db["failure_fingerprints"].insert(row, pk="fingerprint_id")
        return row

    new_count = existing["occurrence_count"] + 1
    stored_fix = existing.get("example_fix") or ""
    effective_fix = stored_fix or fix or ""
    updates: dict[str, Any] = {
        "last_seen_at": now,
        "occurrence_count": new_count,
        "confidence": _confidence(new_count, bool(effective_fix)),
    }
    # Write fix only if not already stored
    if fix and not stored_fix:
        updates["example_fix"] = fix
    db["failure_fingerprints"].update(fp_id, updates)
    return {**existing, **updates}


def update_fingerprint_fix(db: Database, fp_id: str, fix: str) -> None:
    """Store a verified fix for an existing fingerprint and recompute confidence.

    Overwrites any previously stored fix.  Typically called after AI analysis
    confirms a fix works via DVR replay.
    """
    if not db["failure_fingerprints"].exists():
        return
    try:
        row = db["failure_fingerprints"].get(fp_id)
    except Exception:
        return
    if row is None:
        return
    db["failure_fingerprints"].update(
        fp_id,
        {
            "example_fix": fix,
            "confidence": _confidence(row["occurrence_count"], True),
        },
    )


def list_fingerprints(db: Database) -> list[dict[str, Any]]:
    """Return all fingerprints sorted by ``occurrence_count`` descending."""
    if not db["failure_fingerprints"].exists():
        return []
    return [
        dict(row)
        for row in db["failure_fingerprints"].rows_where(
            order_by="occurrence_count desc"
        )
    ]


def print_fingerprint_match(fp_id: str, record: dict[str, Any]) -> None:
    """Print the standard fingerprint-match banner to stdout.

    Output format::

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ⚡ [AgentAutopsy] Known Failure Pattern Detected
        ▶ This failure matches a known pattern (FP-7a3f9c)
        ▶ Seen 12 time(s) before.
        ▶ Error type: TimeoutError
        ▶ Fix: Increase the request timeout to 60 s.
        ▶ Confidence: high
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    count = record.get("occurrence_count", 1)
    confidence = record.get("confidence", "low")
    fix = record.get("example_fix", "")
    error_type = record.get("error_type", "")

    sep = "━" * 60
    confidence_color = (
        _ANSI_GREEN
        if confidence == "high"
        else (_ANSI_YELLOW if confidence == "medium" else _ANSI_CYAN)
    )

    print(f"\n{_ANSI_CYAN}{sep}{_ANSI_RESET}")
    print(
        f"{_ANSI_BOLD}{_ANSI_YELLOW}⚡ [AgentAutopsy] Known Failure Pattern Detected"
        f"{_ANSI_RESET}"
    )
    print(
        f"{_ANSI_CYAN}▶ This failure matches a known pattern "
        f"({_ANSI_BOLD}{fp_id}{_ANSI_RESET}{_ANSI_CYAN}){_ANSI_RESET}"
    )
    print(
        f"{_ANSI_CYAN}▶ Seen {_ANSI_BOLD}{count}{_ANSI_RESET}"
        f"{_ANSI_CYAN} time(s) before.{_ANSI_RESET}"
    )
    if error_type:
        print(f"{_ANSI_CYAN}▶ Error type: {_ANSI_BOLD}{error_type}{_ANSI_RESET}")
    if fix:
        print(f"{_ANSI_CYAN}▶ Fix: {_ANSI_GREEN}{fix}{_ANSI_RESET}")
    print(
        f"{_ANSI_CYAN}▶ Confidence: "
        f"{confidence_color}{_ANSI_BOLD}{confidence}{_ANSI_RESET}"
    )
    print(f"{_ANSI_CYAN}{sep}{_ANSI_RESET}\n")
