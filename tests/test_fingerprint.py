"""Tests for failure DNA fingerprinting.

Covers:
- Fingerprint generation consistency (same structure → same ID)
- Normalization of volatile tokens
- Fingerprint matching across repeated failures
- Occurrence count tracking and confidence escalation
- Fix storage and update behaviour
- list_fingerprints sort order
- patterns CLI command output
- print_fingerprint_match banner content
"""
from __future__ import annotations

import io
import sys
import unittest
from unittest.mock import patch

from sqlite_utils import Database

from agentautopsy.fingerprint import (
    _confidence,
    _find_failing_tool,
    _step_position,
    ensure_fingerprint_tables,
    generate_fingerprint,
    list_fingerprints,
    match_fingerprint,
    normalize_root_cause,
    print_fingerprint_match,
    record_fingerprint,
    update_fingerprint_fix,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_EVENTS_TIMEOUT = [
    {"type": "llm_call", "payload": {"model": "gpt-4", "messages": []}},
    {"type": "tool_call", "payload": {"tool": "get_weather", "args": {}}},
    {
        "type": "error",
        "payload": {"error_type": "TimeoutError", "message": "request timed out after 30s"},
    },
]

_FAILURE_TIMEOUT = {
    "failed": True,
    "error_type": "TimeoutError",
    "message": "request timed out after 30s",
    "run_id": "abc-123-def-456-000000000000",
    "failure_event_id": "evt-abc123",
}

_TRACE_TIMEOUT = {"failure": _FAILURE_TIMEOUT, "events": _EVENTS_TIMEOUT}


def _fresh_db() -> Database:
    """Return a new in-memory Database with the fingerprint table created."""
    db = Database(memory=True)
    ensure_fingerprint_tables(db)
    return db


# ---------------------------------------------------------------------------
# normalize_root_cause
# ---------------------------------------------------------------------------

class TestNormalizeRootCause(unittest.TestCase):
    def test_strips_uuid(self) -> None:
        text = "run 550e8400-e29b-41d4-a716-446655440000 failed"
        result = normalize_root_cause(text)
        self.assertNotIn("550e8400", result)
        self.assertIn("<id>", result)

    def test_strips_iso_timestamp(self) -> None:
        text = "failed at 2024-01-15T12:34:56Z during processing"
        result = normalize_root_cause(text)
        self.assertNotIn("2024-01-15", result)
        self.assertIn("<timestamp>", result)

    def test_strips_url(self) -> None:
        text = "cannot connect to https://api.openai.com/v1/chat"
        result = normalize_root_cause(text)
        self.assertNotIn("api.openai.com", result)
        self.assertIn("<url>", result)

    def test_strips_standalone_numbers(self) -> None:
        text = "timed out after 30 seconds on attempt 3"
        result = normalize_root_cause(text)
        # Volatile numbers should be replaced
        self.assertNotIn(" 30 ", result)
        self.assertNotIn(" 3", result)

    def test_lowercases_output(self) -> None:
        text = "TimeoutError: Connection Reset By Peer"
        result = normalize_root_cause(text)
        self.assertEqual(result, result.lower())

    def test_idempotent(self) -> None:
        text = "request timed out after 30s"
        once = normalize_root_cause(text)
        twice = normalize_root_cause(once)
        self.assertEqual(once, twice)

    def test_collapses_whitespace(self) -> None:
        text = "error  in   step    5"
        result = normalize_root_cause(text)
        self.assertNotIn("  ", result)

    def test_different_uuids_same_pattern(self) -> None:
        a = normalize_root_cause("run 550e8400-e29b-41d4-a716-446655440000 failed")
        b = normalize_root_cause("run deadbeef-dead-beef-dead-beefdeadbeef failed")
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestFindFailingTool(unittest.TestCase):
    def test_returns_last_tool_before_error(self) -> None:
        events = [
            {"type": "tool_call", "payload": {"tool": "search"}},
            {"type": "tool_call", "payload": {"tool": "get_weather"}},
            {"type": "error", "payload": {"error_type": "TimeoutError"}},
        ]
        self.assertEqual(_find_failing_tool(events), "get_weather")

    def test_falls_back_when_no_tool_call(self) -> None:
        events = [
            {"type": "llm_call", "payload": {}},
            {"type": "error", "payload": {"error_type": "TimeoutError"}},
        ]
        self.assertEqual(_find_failing_tool(events), "unknown_tool")

    def test_tool_after_error_is_ignored(self) -> None:
        events = [
            {"type": "error", "payload": {}},
            {"type": "tool_call", "payload": {"tool": "late_tool"}},
        ]
        self.assertEqual(_find_failing_tool(events), "unknown_tool")

    def test_payload_as_json_string(self) -> None:
        import json
        events = [
            {"type": "tool_call", "payload": json.dumps({"tool": "summarize"})},
            {"type": "error", "payload": {}},
        ]
        self.assertEqual(_find_failing_tool(events), "summarize")


class TestStepPosition(unittest.TestCase):
    def test_early(self) -> None:
        self.assertEqual(_step_position(0, 9), "early")
        self.assertEqual(_step_position(2, 9), "early")

    def test_mid(self) -> None:
        self.assertEqual(_step_position(4, 9), "mid")
        self.assertEqual(_step_position(5, 9), "mid")

    def test_late(self) -> None:
        self.assertEqual(_step_position(7, 9), "late")
        self.assertEqual(_step_position(9, 9), "late")

    def test_zero_total(self) -> None:
        self.assertEqual(_step_position(0, 0), "early")


class TestConfidence(unittest.TestCase):
    def test_low_when_rare_and_no_fix(self) -> None:
        self.assertEqual(_confidence(1, False), "low")
        self.assertEqual(_confidence(2, False), "low")

    def test_medium_with_fix_at_two(self) -> None:
        self.assertEqual(_confidence(2, True), "medium")

    def test_medium_at_three(self) -> None:
        self.assertEqual(_confidence(3, False), "medium")

    def test_high_requires_five_and_fix(self) -> None:
        self.assertEqual(_confidence(5, True), "high")
        # 5 occurrences without a fix is still medium
        self.assertEqual(_confidence(5, False), "medium")
        # 4 with fix is still medium
        self.assertEqual(_confidence(4, True), "medium")


# ---------------------------------------------------------------------------
# generate_fingerprint — consistency and discrimination
# ---------------------------------------------------------------------------

class TestGenerateFingerprint(unittest.TestCase):
    def test_returns_fp_prefix(self) -> None:
        fp = generate_fingerprint(_TRACE_TIMEOUT)
        self.assertTrue(fp.startswith("FP-"), fp)

    def test_length_is_nine_chars(self) -> None:
        fp = generate_fingerprint(_TRACE_TIMEOUT)
        # "FP-" + 6 hex chars = 9
        self.assertEqual(len(fp), 9)

    def test_deterministic_same_trace(self) -> None:
        fp1 = generate_fingerprint(_TRACE_TIMEOUT)
        fp2 = generate_fingerprint(_TRACE_TIMEOUT)
        self.assertEqual(fp1, fp2)

    def test_same_structure_different_volatile_tokens_same_id(self) -> None:
        """UUIDs and timestamps in the message must not change the fingerprint."""
        trace_a = {
            "failure": {"error_type": "TimeoutError", "message": "timed out after 30s"},
            "events": _EVENTS_TIMEOUT,
        }
        trace_b = {
            "failure": {
                "error_type": "TimeoutError",
                # Different UUID and timestamp in the message
                "message": (
                    "timed out after 30s run=550e8400-e29b-41d4-a716-446655440000"
                    " at 2025-06-01T10:00:00Z"
                ),
            },
            "events": _EVENTS_TIMEOUT,
        }
        self.assertEqual(generate_fingerprint(trace_a), generate_fingerprint(trace_b))

    def test_different_error_type_gives_different_id(self) -> None:
        trace_timeout = {
            "failure": {"error_type": "TimeoutError", "message": "request timed out"},
            "events": _EVENTS_TIMEOUT,
        }
        trace_conn = {
            "failure": {"error_type": "APIConnectionError", "message": "request timed out"},
            "events": _EVENTS_TIMEOUT,
        }
        self.assertNotEqual(
            generate_fingerprint(trace_timeout), generate_fingerprint(trace_conn)
        )

    def test_different_failing_tool_gives_different_id(self) -> None:
        events_weather = [
            {"type": "tool_call", "payload": {"tool": "get_weather"}},
            {"type": "error", "payload": {"error_type": "TimeoutError", "message": "timeout"}},
        ]
        events_stocks = [
            {"type": "tool_call", "payload": {"tool": "get_stocks"}},
            {"type": "error", "payload": {"error_type": "TimeoutError", "message": "timeout"}},
        ]
        trace_w = {"failure": {"error_type": "TimeoutError", "message": "timeout"}, "events": events_weather}
        trace_s = {"failure": {"error_type": "TimeoutError", "message": "timeout"}, "events": events_stocks}
        self.assertNotEqual(generate_fingerprint(trace_w), generate_fingerprint(trace_s))

    def test_flat_trace_layout_accepted(self) -> None:
        """Top-level error_type/message keys (no nested 'failure') must work."""
        flat = {
            "error_type": "TimeoutError",
            "message": "request timed out after 30s",
            "events": _EVENTS_TIMEOUT,
        }
        fp = generate_fingerprint(flat)
        self.assertTrue(fp.startswith("FP-"))

    def test_empty_events_does_not_raise(self) -> None:
        trace = {"failure": {"error_type": "ValueError", "message": "bad value"}, "events": []}
        fp = generate_fingerprint(trace)
        self.assertTrue(fp.startswith("FP-"))

    def test_missing_error_type_handled(self) -> None:
        trace = {"failure": {"message": "something broke"}, "events": []}
        fp = generate_fingerprint(trace)
        self.assertTrue(fp.startswith("FP-"))


# ---------------------------------------------------------------------------
# Fingerprint matching across repeated failures
# ---------------------------------------------------------------------------

class TestFingerprintMatching(unittest.TestCase):
    def test_no_match_on_empty_db(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        self.assertIsNone(match_fingerprint(db, fp_id))

    def test_match_returned_after_first_record(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        match = match_fingerprint(db, fp_id)
        self.assertIsNotNone(match)
        self.assertEqual(match["fingerprint_id"], fp_id)
        self.assertEqual(match["occurrence_count"], 1)

    def test_repeated_failures_increment_occurrence_count(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        for _ in range(4):
            record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        match = match_fingerprint(db, fp_id)
        self.assertEqual(match["occurrence_count"], 4)

    def test_first_seen_at_preserved_across_updates(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        first = record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        first_seen = first["first_seen_at"]
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        match = match_fingerprint(db, fp_id)
        self.assertEqual(match["first_seen_at"], first_seen)

    def test_last_seen_at_updated_on_repeat(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        first = record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        match = match_fingerprint(db, fp_id)
        # last_seen_at must be >= first_seen_at
        self.assertGreaterEqual(match["last_seen_at"], first["first_seen_at"])

    def test_fix_stored_on_first_record(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT, fix="Increase timeout to 60s.")
        match = match_fingerprint(db, fp_id)
        self.assertEqual(match["example_fix"], "Increase timeout to 60s.")

    def test_fix_not_overwritten_by_later_record(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT, fix="Original fix")
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT, fix="Replacement fix")
        match = match_fingerprint(db, fp_id)
        self.assertEqual(match["example_fix"], "Original fix")

    def test_fix_written_on_later_record_if_not_yet_stored(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)          # no fix
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT, fix="Late fix")  # first fix
        match = match_fingerprint(db, fp_id)
        self.assertEqual(match["example_fix"], "Late fix")

    def test_update_fingerprint_fix_overwrites(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT, fix="Old fix")
        update_fingerprint_fix(db, fp_id, "Definitive fix")
        match = match_fingerprint(db, fp_id)
        self.assertEqual(match["example_fix"], "Definitive fix")

    def test_update_fingerprint_fix_noop_when_table_missing(self) -> None:
        db = Database(memory=True)   # no table
        # Must not raise
        update_fingerprint_fix(db, "FP-aabbcc", "some fix")

    def test_confidence_low_on_first_occurrence(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        rec = record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        self.assertEqual(rec["confidence"], "low")

    def test_confidence_escalates_to_high(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        for _ in range(5):
            record_fingerprint(db, fp_id, _TRACE_TIMEOUT, fix="some fix")
        match = match_fingerprint(db, fp_id)
        self.assertEqual(match["confidence"], "high")

    def test_error_type_persisted(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        match = match_fingerprint(db, fp_id)
        self.assertEqual(match["error_type"], "TimeoutError")

    def test_failing_tool_persisted(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        match = match_fingerprint(db, fp_id)
        self.assertEqual(match["failing_tool"], "get_weather")

    def test_step_position_persisted(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        match = match_fingerprint(db, fp_id)
        self.assertIn(match["step_position"], ("early", "mid", "late"))


# ---------------------------------------------------------------------------
# list_fingerprints — sort order
# ---------------------------------------------------------------------------

class TestListFingerprints(unittest.TestCase):
    def test_empty_list_when_no_records(self) -> None:
        db = _fresh_db()
        self.assertEqual(list_fingerprints(db), [])

    def test_empty_list_when_table_missing(self) -> None:
        db = Database(memory=True)
        self.assertEqual(list_fingerprints(db), [])

    def test_sorted_by_occurrence_count_descending(self) -> None:
        db = _fresh_db()

        trace_a = {
            "failure": {"error_type": "TimeoutError", "message": "timeout"},
            "events": [{"type": "error", "payload": {}}],
        }
        trace_b = {
            "failure": {"error_type": "ValueError", "message": "bad value"},
            "events": [{"type": "error", "payload": {}}],
        }
        fp_a = generate_fingerprint(trace_a)
        fp_b = generate_fingerprint(trace_b)

        record_fingerprint(db, fp_a, trace_a)           # 1 occurrence
        for _ in range(3):
            record_fingerprint(db, fp_b, trace_b)       # 3 occurrences

        results = list_fingerprints(db)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["fingerprint_id"], fp_b)
        self.assertEqual(results[1]["fingerprint_id"], fp_a)

    def test_all_fields_present(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        results = list_fingerprints(db)
        required = {
            "fingerprint_id", "error_type", "failing_tool", "step_position",
            "first_seen_at", "last_seen_at", "occurrence_count",
            "example_root_cause", "example_fix", "confidence",
        }
        self.assertEqual(required, required & results[0].keys())


# ---------------------------------------------------------------------------
# patterns CLI command (inline simulation matching cli.py logic)
# ---------------------------------------------------------------------------

def _run_patterns_output(db: Database) -> str:
    """Simulate what ``agentautopsy patterns`` prints and capture it."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        records = list_fingerprints(db)
        if not records:
            print("No failure patterns recorded yet.")
        else:
            print(
                f"\n{'FINGERPRINT':<12} {'OCCURRENCES':>11} {'CONFIDENCE':<12} "
                f"{'ERROR TYPE':<25} {'TOOL':<20} {'POSITION':<8}"
            )
            print("─" * 100)
            for rec in records:
                fp_id = rec["fingerprint_id"]
                count = rec["occurrence_count"]
                confidence = rec["confidence"]
                error_type = (rec.get("error_type") or "")[:24]
                tool = (rec.get("failing_tool") or "")[:19]
                position = rec.get("step_position") or ""
                has_fix = "✓" if rec.get("example_fix") else " "
                print(
                    f"{fp_id:<12} {count:>11}   {confidence:<12} "
                    f"{error_type:<25} {tool:<20} {position:<8} {has_fix}"
                )
            print()
            total = sum(r["occurrence_count"] for r in records)
            print(f"{len(records)} pattern(s) · {total} total occurrence(s)")
    return buf.getvalue()


class TestPatternsCLICommand(unittest.TestCase):
    def test_empty_message_when_no_patterns(self) -> None:
        db = _fresh_db()
        output = _run_patterns_output(db)
        self.assertIn("No failure patterns recorded yet.", output)

    def test_fingerprint_id_in_output(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        output = _run_patterns_output(db)
        self.assertIn(fp_id, output)

    def test_occurrence_count_in_output(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        for _ in range(3):
            record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        output = _run_patterns_output(db)
        self.assertIn("3", output)

    def test_summary_line_shows_pattern_count(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT)
        output = _run_patterns_output(db)
        self.assertIn("1 pattern(s)", output)
        self.assertIn("2 total occurrence(s)", output)

    def test_fix_checkmark_when_fix_stored(self) -> None:
        db = _fresh_db()
        fp_id = generate_fingerprint(_TRACE_TIMEOUT)
        record_fingerprint(db, fp_id, _TRACE_TIMEOUT, fix="Increase timeout")
        output = _run_patterns_output(db)
        self.assertIn("✓", output)

    def test_multiple_patterns_all_shown(self) -> None:
        db = _fresh_db()
        for error_type in ("TimeoutError", "ValueError", "APIConnectionError"):
            trace = {
                "failure": {"error_type": error_type, "message": "something failed"},
                "events": [{"type": "error", "payload": {}}],
            }
            fp_id = generate_fingerprint(trace)
            record_fingerprint(db, fp_id, trace)
        output = _run_patterns_output(db)
        self.assertIn("3 pattern(s)", output)


# ---------------------------------------------------------------------------
# print_fingerprint_match — output format
# ---------------------------------------------------------------------------

class TestPrintFingerprintMatch(unittest.TestCase):
    def _capture_match(self, fp_id: str, record: dict) -> str:
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_fingerprint_match(fp_id, record)
        return buf.getvalue()

    def test_contains_fingerprint_id(self) -> None:
        record = {
            "fingerprint_id": "FP-abc123",
            "occurrence_count": 5,
            "confidence": "medium",
            "error_type": "TimeoutError",
            "example_fix": "Increase timeout to 60s.",
        }
        output = self._capture_match("FP-abc123", record)
        self.assertIn("FP-abc123", output)

    def test_contains_occurrence_count(self) -> None:
        record = {
            "fingerprint_id": "FP-abc123",
            "occurrence_count": 12,
            "confidence": "high",
            "error_type": "TimeoutError",
            "example_fix": "Some fix",
        }
        output = self._capture_match("FP-abc123", record)
        self.assertIn("12", output)

    def test_contains_confidence_label(self) -> None:
        record = {
            "fingerprint_id": "FP-abc123",
            "occurrence_count": 7,
            "confidence": "high",
            "error_type": "TimeoutError",
            "example_fix": "Some fix",
        }
        output = self._capture_match("FP-abc123", record)
        self.assertIn("high", output)

    def test_contains_fix_text(self) -> None:
        record = {
            "fingerprint_id": "FP-xyz789",
            "occurrence_count": 3,
            "confidence": "medium",
            "error_type": "ValueError",
            "example_fix": "Validate the schema before calling the tool.",
        }
        output = self._capture_match("FP-xyz789", record)
        self.assertIn("Validate the schema before calling the tool.", output)

    def test_fix_line_omitted_when_empty(self) -> None:
        record = {
            "fingerprint_id": "FP-nofix1",
            "occurrence_count": 1,
            "confidence": "low",
            "error_type": "TimeoutError",
            "example_fix": "",
        }
        output = self._capture_match("FP-nofix1", record)
        # Should not print a "Fix:" line when there is no fix
        self.assertNotIn("Fix:", output)


if __name__ == "__main__":
    unittest.main()
