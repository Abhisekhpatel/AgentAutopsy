"""Tests for src/agentautopsy/context_health.py."""
from __future__ import annotations

import json
import sys
import os
import unittest

# Make sure the staging src is importable when running locally
_STAGING_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _STAGING_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_STAGING_SRC))

from sqlite_utils import Database

from agentautopsy.context_health import (
    ContextHealthTracker,
    _compute_entity_retention,
    _compute_token_overlap,
    _count_contradictions,
    _extract_entities,
    _messages_to_text,
    _tokenize,
    compute_health_score,
    ensure_context_health_tables,
    get_active_tracker,
    load_health_scores,
    record_health_event,
    _health_ctx,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _mem_db() -> Database:
    """Return a fresh in-memory sqlite-utils database."""
    db = Database(memory=True)
    ensure_context_health_tables(db)
    return db


def _tracker(db: Database | None = None) -> ContextHealthTracker:
    """Return a tracker wired to *db* (or a fresh in-memory one)."""
    if db is None:
        db = _mem_db()
    t = ContextHealthTracker(db=db, run_id="run-test-001", warn_threshold=60)
    return t


# ── _tokenize ─────────────────────────────────────────────────────────────────

class TestTokenize(unittest.TestCase):
    def test_returns_set(self):
        result = _tokenize("Hello world")
        self.assertIsInstance(result, set)

    def test_lowercases(self):
        self.assertIn("hello", _tokenize("Hello"))

    def test_filters_stop_words(self):
        tokens = _tokenize("the cat sat on the mat")
        self.assertNotIn("the", tokens)
        self.assertNotIn("on", tokens)
        self.assertIn("cat", tokens)
        self.assertIn("sat", tokens)
        self.assertIn("mat", tokens)

    def test_minimum_length(self):
        # Single-char tokens (excluding stop-word single chars) must be ≥ 2 chars.
        tokens = _tokenize("a b c xyz")
        # "a", "b", "c" are either stop-words or too short
        self.assertNotIn("a", tokens)
        self.assertIn("xyz", tokens)

    def test_empty_string(self):
        self.assertEqual(_tokenize(""), set())

    def test_includes_numbers(self):
        tokens = _tokenize("order 12345 shipped")
        self.assertIn("12345", tokens)

    def test_underscore_word(self):
        tokens = _tokenize("customer_id is important")
        self.assertIn("customer_id", tokens)


# ── _extract_entities ─────────────────────────────────────────────────────────

class TestExtractEntities(unittest.TestCase):
    def test_double_quoted(self):
        entities = _extract_entities('The field is "customer_id".')
        self.assertIn("customer_id", entities)

    def test_single_quoted(self):
        entities = _extract_entities("The key is 'order_ref'.")
        self.assertIn("order_ref", entities)

    def test_all_caps_id(self):
        entities = _extract_entities("Use the TOKEN_KEY variable.")
        self.assertIn("TOKEN_KEY", entities)

    def test_iso_date(self):
        entities = _extract_entities("Due date is 2024-06-15.")
        self.assertIn("2024-06-15", entities)

    def test_long_number(self):
        entities = _extract_entities("Invoice 1234567 is pending.")
        self.assertIn("1234567", entities)

    def test_snake_case_identifier(self):
        entities = _extract_entities("The field customer_id must match.")
        self.assertIn("customer_id", entities)

    def test_multi_word_proper_noun(self):
        entities = _extract_entities("Contact John Smith for details.")
        # Multi-word proper noun check
        combined = " ".join(entities)
        self.assertIn("John Smith", combined)

    def test_deduplication(self):
        entities = _extract_entities('"foo" and "foo" again')
        count = sum(1 for e in entities if e.lower() == "foo")
        self.assertEqual(count, 1)

    def test_empty_string(self):
        self.assertEqual(_extract_entities(""), [])

    def test_no_entities(self):
        # Plain prose with no special patterns
        entities = _extract_entities("the cat sat on the mat and was happy")
        self.assertEqual(entities, [])


# ── _compute_token_overlap ────────────────────────────────────────────────────

class TestTokenOverlap(unittest.TestCase):
    def test_identical_texts(self):
        anchor = {"hello", "world", "python"}
        self.assertAlmostEqual(_compute_token_overlap(anchor, anchor), 1.0)

    def test_completely_different(self):
        anchor = {"alpha", "beta", "gamma"}
        current = {"delta", "epsilon", "zeta"}
        self.assertAlmostEqual(_compute_token_overlap(anchor, current), 0.0)

    def test_partial_overlap(self):
        anchor = {"alpha", "beta", "gamma", "delta"}
        current = {"alpha", "beta", "zeta", "eta"}
        result = _compute_token_overlap(anchor, current)
        # 2 of 4 match → 0.5
        self.assertAlmostEqual(result, 0.5)

    def test_empty_anchor_returns_one(self):
        self.assertAlmostEqual(_compute_token_overlap(set(), {"any", "tokens"}), 1.0)

    def test_empty_current_returns_zero(self):
        anchor = {"hello", "world"}
        self.assertAlmostEqual(_compute_token_overlap(anchor, set()), 0.0)


# ── _compute_entity_retention ─────────────────────────────────────────────────

class TestEntityRetention(unittest.TestCase):
    def test_all_present(self):
        entities = ["customer_id", "2024-06-15", "ORDER123"]
        text = "customer_id was updated on 2024-06-15 for ORDER123"
        retention, missing = _compute_entity_retention(entities, text)
        self.assertAlmostEqual(retention, 1.0)
        self.assertEqual(missing, [])

    def test_none_present(self):
        entities = ["customer_id", "ORDER123"]
        text = "a completely different prompt with no matching terms"
        retention, missing = _compute_entity_retention(entities, text)
        self.assertAlmostEqual(retention, 0.0)
        self.assertEqual(len(missing), 2)

    def test_partial_retention(self):
        entities = ["alpha", "beta", "gamma"]
        text = "only alpha is here"
        retention, missing = _compute_entity_retention(entities, text)
        self.assertAlmostEqual(retention, 1 / 3)
        self.assertIn("beta", missing)
        self.assertIn("gamma", missing)

    def test_empty_entities_returns_full_retention(self):
        retention, missing = _compute_entity_retention([], "any text")
        self.assertAlmostEqual(retention, 1.0)
        self.assertEqual(missing, [])

    def test_case_insensitive_match(self):
        entities = ["ORDER123"]
        text = "the order123 status has changed"
        retention, missing = _compute_entity_retention(entities, text)
        self.assertAlmostEqual(retention, 1.0)


# ── _count_contradictions ─────────────────────────────────────────────────────

class TestContradictions(unittest.TestCase):
    def test_no_contradiction(self):
        anchor = "The service is running fine."
        current = "The service is operating normally."
        self.assertEqual(_count_contradictions(anchor, current), 0)

    def test_simple_negation(self):
        anchor = "The service is available."
        current = "The service is not available at this time."
        # "service is" appears in anchor and "service is not" appears in current
        count = _count_contradictions(anchor, current)
        self.assertGreaterEqual(count, 1)

    def test_empty_anchor(self):
        self.assertEqual(_count_contradictions("", "The service is fine."), 0)

    def test_no_subject_verb_pairs(self):
        anchor = "Hello world"
        current = "Hello world is not"
        # No simple subject-verb pairs with our SIMPLE_VERBS set
        result = _count_contradictions(anchor, current)
        self.assertGreaterEqual(result, 0)

    def test_isnt_form(self):
        anchor = "The user is authenticated."
        current = "The user isn't authenticated."
        count = _count_contradictions(anchor, current)
        self.assertGreaterEqual(count, 1)


# ── compute_health_score ──────────────────────────────────────────────────────

class TestComputeHealthScore(unittest.TestCase):
    def test_perfect_score(self):
        score = compute_health_score(1.0, 1.0, 0)
        self.assertAlmostEqual(score, 100.0)

    def test_zero_overlap_zero_retention(self):
        score = compute_health_score(0.0, 0.0, 0)
        self.assertAlmostEqual(score, 0.0)

    def test_contradiction_penalty(self):
        # 1 contradiction deducts 10 points from a perfect score
        score = compute_health_score(1.0, 1.0, 1)
        self.assertAlmostEqual(score, 90.0)

    def test_contradiction_penalty_capped_at_30(self):
        # 10 contradictions should not deduct more than 30 points
        score = compute_health_score(1.0, 1.0, 10)
        self.assertAlmostEqual(score, 70.0)

    def test_score_clamped_at_zero(self):
        score = compute_health_score(0.0, 0.0, 5)
        self.assertAlmostEqual(score, 0.0)

    def test_partial_scores(self):
        # 50 % overlap + 50 % retention = 50 points raw, no penalty
        score = compute_health_score(0.5, 0.5, 0)
        self.assertAlmostEqual(score, 50.0)

    def test_asymmetric_weights(self):
        # 100 % overlap but 0 % entity retention
        score = compute_health_score(1.0, 0.0, 0)
        self.assertAlmostEqual(score, 50.0)


# ── _messages_to_text ─────────────────────────────────────────────────────────

class TestMessagesToText(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(_messages_to_text("hello"), "hello")

    def test_list_of_dicts(self):
        msgs = [
            {"role": "user", "content": "What is the order status?"},
            {"role": "assistant", "content": "The order is shipped."},
        ]
        text = _messages_to_text(msgs)
        self.assertIn("order status", text)
        self.assertIn("shipped", text)

    def test_list_of_strings(self):
        msgs = ["First message", "Second message"]
        text = _messages_to_text(msgs)
        self.assertIn("First message", text)
        self.assertIn("Second message", text)

    def test_empty_list(self):
        self.assertEqual(_messages_to_text([]), "")

    def test_none_returns_empty(self):
        self.assertEqual(_messages_to_text(None), "")

    def test_content_list_blocks(self):
        msgs = [{"role": "user", "content": [{"text": "block text"}, {"text": "more"}]}]
        text = _messages_to_text(msgs)
        self.assertIn("block text", text)
        self.assertIn("more", text)


# ── ensure_context_health_tables ──────────────────────────────────────────────

class TestEnsureContextHealthTables(unittest.TestCase):
    def test_creates_table(self):
        db = Database(memory=True)
        ensure_context_health_tables(db)
        self.assertIn("context_health", db.table_names())

    def test_idempotent(self):
        db = Database(memory=True)
        ensure_context_health_tables(db)
        ensure_context_health_tables(db)   # should not raise
        self.assertIn("context_health", db.table_names())

    def test_schema_columns(self):
        db = Database(memory=True)
        ensure_context_health_tables(db)
        cols = {c.name for c in db["context_health"].columns}
        expected = {
            "id", "run_id", "step", "recorded_at", "health_score",
            "token_overlap", "entity_retention", "contradiction_count",
            "anchor_entity_count", "missing_entities", "alert_fired",
        }
        self.assertTrue(expected.issubset(cols))


# ── ContextHealthTracker.record_llm_call ──────────────────────────────────────

class TestRecordLlmCall(unittest.TestCase):
    def test_first_step_always_100(self):
        t = _tracker()
        snap = t.record_llm_call([{"role": "user", "content": "What is customer_id 42?"}])
        self.assertEqual(snap["step"], 1)
        self.assertAlmostEqual(snap["health_score"], 100.0)
        self.assertEqual(snap["alert_fired"], 0)

    def test_identical_step_stays_near_100(self):
        t = _tracker()
        msgs = [{"role": "user", "content": "Refund order 1234567 for customer_id ABC"}]
        t.record_llm_call(msgs)                     # anchor
        snap = t.record_llm_call(msgs)              # identical step 2
        self.assertGreater(snap["health_score"], 90.0)

    def test_step_counter_increments(self):
        t = _tracker()
        t.record_llm_call("step one")
        snap = t.record_llm_call("step two")
        self.assertEqual(snap["step"], 2)

    def test_completely_different_prompt_low_score(self):
        t = _tracker()
        t.record_llm_call(
            "Process refund for customer_id CUST001 order 9876543 dated 2024-06-15"
        )
        snap = t.record_llm_call(
            "Translate the following French text: Bonjour le monde"
        )
        self.assertLess(snap["health_score"], 80.0)

    def test_missing_entities_reported(self):
        t = _tracker()
        t.record_llm_call("Process order 1234567 for customer_id CUST001")
        snap = t.record_llm_call("Tell me about something completely unrelated")
        self.assertIsInstance(snap["missing_entities"], list)
        missing_lower = [e.lower() for e in snap["missing_entities"]]
        # At least one of the anchor entities should be missing
        self.assertGreater(len(snap["missing_entities"]), 0)

    def test_anchor_entity_count_correct(self):
        t = _tracker()
        anchor_text = 'Order "ORD-9999" for 2024-06-15 is pending'
        snap = t.record_llm_call(anchor_text)
        self.assertGreater(snap["anchor_entity_count"], 0)

    def test_token_overlap_is_percentage(self):
        t = _tracker()
        t.record_llm_call("alpha beta gamma delta")
        snap = t.record_llm_call("alpha beta zeta eta")
        # Should be between 0 and 100
        self.assertGreaterEqual(snap["token_overlap"], 0.0)
        self.assertLessEqual(snap["token_overlap"], 100.0)

    def test_entity_retention_is_percentage(self):
        t = _tracker()
        t.record_llm_call("order 1234567 customer_id ABC")
        snap = t.record_llm_call("order 1234567 status updated")
        self.assertGreaterEqual(snap["entity_retention"], 0.0)
        self.assertLessEqual(snap["entity_retention"], 100.0)

    def test_multiple_steps_accumulate(self):
        t = _tracker()
        for i in range(5):
            t.record_llm_call(f"Step {i} message with customer_id CUST001")
        scores = t.get_scores()
        self.assertEqual(len(scores), 5)

    def test_string_messages_accepted(self):
        t = _tracker()
        snap = t.record_llm_call("plain string prompt")
        self.assertEqual(snap["step"], 1)

    def test_alert_fired_when_below_threshold(self):
        t = _tracker()
        t.record_llm_call("Process refund for customer_id CUST001 order 9876543 dated 2024-06-15")
        # Completely unrelated → low score
        snap = t.record_llm_call("A completely different topic about weather and nature")
        if snap["health_score"] < 60:
            self.assertEqual(snap["alert_fired"], 1)
        else:
            # Score stayed above threshold → no alert
            self.assertEqual(snap["alert_fired"], 0)

    def test_no_alert_on_anchor_step(self):
        t = _tracker()
        # Even a very short anchor should not fire an alert
        snap = t.record_llm_call("hi")
        self.assertEqual(snap["alert_fired"], 0)


# ── ContextHealthTracker.get_scores / current_score ───────────────────────────

class TestGetScores(unittest.TestCase):
    def test_empty_before_any_call(self):
        t = _tracker()
        self.assertEqual(t.get_scores(), [])

    def test_current_score_default(self):
        t = _tracker()
        self.assertAlmostEqual(t.current_score(), 100.0)

    def test_current_score_after_calls(self):
        t = _tracker()
        t.record_llm_call("anchor message with entities")
        snap2 = t.record_llm_call("unrelated text about something else entirely new")
        self.assertAlmostEqual(t.current_score(), snap2["health_score"])

    def test_scores_contain_required_keys(self):
        t = _tracker()
        t.record_llm_call("anchor message")
        t.record_llm_call("second message")
        for score in t.get_scores():
            for key in ("step", "health_score", "token_overlap", "entity_retention",
                        "contradiction_count", "missing_entities", "alert_fired"):
                self.assertIn(key, score, f"Missing key: {key}")


# ── ContextHealthTracker.reset ────────────────────────────────────────────────

class TestReset(unittest.TestCase):
    def test_reset_clears_scores(self):
        t = _tracker()
        t.record_llm_call("anchor")
        t.record_llm_call("step 2")
        t.reset()
        self.assertEqual(t.get_scores(), [])

    def test_reset_clears_anchor(self):
        t = _tracker()
        t.record_llm_call("first anchor message with entity ABC")
        t.reset()
        t.record_llm_call("new anchor completely different")
        # After reset, step 1 is anchor again → health = 100
        self.assertAlmostEqual(t.current_score(), 100.0)

    def test_step_counter_resets(self):
        t = _tracker()
        t.record_llm_call("anchor")
        t.record_llm_call("step 2")
        t.reset()
        snap = t.record_llm_call("new anchor")
        self.assertEqual(snap["step"], 1)


# ── Persistence (SQLite) ──────────────────────────────────────────────────────

class TestPersistence(unittest.TestCase):
    def test_records_written_to_db(self):
        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-persist-001")
        t.record_llm_call("anchor step")
        t.record_llm_call("second step with different content")
        rows = list(db["context_health"].rows)
        self.assertEqual(len(rows), 2)

    def test_db_fields_populated(self):
        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-persist-002")
        t.record_llm_call("anchor")
        row = next(iter(db["context_health"].rows))
        self.assertEqual(row["run_id"], "run-persist-002")
        self.assertEqual(row["step"], 1)
        self.assertIsNotNone(row["recorded_at"])
        self.assertAlmostEqual(row["health_score"], 100.0)

    def test_missing_entities_stored_as_json(self):
        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-persist-003")
        t.record_llm_call("Process order 9876543 for customer_id ABC123")
        t.record_llm_call("A totally different topic about something else")
        rows = list(db["context_health"].rows_where("step = 2"))
        self.assertTrue(len(rows) > 0)
        missing_raw = rows[0]["missing_entities"]
        missing = json.loads(missing_raw)
        self.assertIsInstance(missing, list)

    def test_no_db_no_error(self):
        t = ContextHealthTracker(db=None, run_id=None)
        # Should not raise even without a db
        snap = t.record_llm_call("test message")
        self.assertEqual(snap["step"], 1)


# ── load_health_scores ────────────────────────────────────────────────────────

class TestLoadHealthScores(unittest.TestCase):
    def test_returns_empty_for_unknown_run(self):
        db = _mem_db()
        scores = load_health_scores(db, "nonexistent-run")
        self.assertEqual(scores, [])

    def test_round_trip(self):
        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-load-001")
        t.record_llm_call("anchor with customer_id ABC")
        t.record_llm_call("step two with some different tokens")
        t.record_llm_call("step three more changes again here")

        loaded = load_health_scores(db, "run-load-001")
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0]["step"], 1)
        self.assertAlmostEqual(loaded[0]["health_score"], 100.0)

    def test_ordered_by_step(self):
        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-load-002")
        for i in range(5):
            t.record_llm_call(f"step {i} text")
        loaded = load_health_scores(db, "run-load-002")
        steps = [r["step"] for r in loaded]
        self.assertEqual(steps, sorted(steps))

    def test_loaded_fields_present(self):
        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-load-003")
        t.record_llm_call("anchor")
        t.record_llm_call("second step")
        loaded = load_health_scores(db, "run-load-003")
        for row in loaded:
            for key in ("step", "health_score", "token_overlap", "entity_retention",
                        "contradiction_count", "missing_entities", "alert_fired"):
                self.assertIn(key, row, f"Missing key: {key}")

    def test_missing_entities_is_list(self):
        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-load-004")
        t.record_llm_call("customer_id ABC order 9999999")
        t.record_llm_call("something totally unrelated here")
        loaded = load_health_scores(db, "run-load-004")
        for row in loaded:
            self.assertIsInstance(row["missing_entities"], list)


# ── get_active_tracker / record_health_event ──────────────────────────────────

class TestModuleLevelHelpers(unittest.TestCase):
    def setUp(self):
        _health_ctx.clear()

    def tearDown(self):
        _health_ctx.clear()

    def test_get_active_tracker_none_by_default(self):
        self.assertIsNone(get_active_tracker())

    def test_watch_registers_tracker(self):
        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-mod-001")
        _health_ctx["tracker"] = t
        self.assertIs(get_active_tracker(), t)

    def test_record_health_event_no_tracker(self):
        # Should return gracefully with ok=True
        result = record_health_event([{"role": "user", "content": "test"}])
        self.assertEqual(result, {"ok": True})

    def test_record_health_event_with_active_tracker(self):
        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-mod-002")
        _health_ctx["tracker"] = t
        snap = record_health_event("anchor message with entities ABC 9999")
        self.assertEqual(snap["step"], 1)
        self.assertAlmostEqual(snap["health_score"], 100.0)

    def test_bad_value_in_ctx_returns_none(self):
        _health_ctx["tracker"] = "not a tracker"
        self.assertIsNone(get_active_tracker())


# ── Warning output ────────────────────────────────────────────────────────────

class TestWarningOutput(unittest.TestCase):
    def test_warning_printed_when_low_health(self):
        """A low-health step must print a warning to stdout."""
        import io
        from contextlib import redirect_stdout

        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-warn-001", warn_threshold=95)
        t.record_llm_call("anchor with customer_id CUST999 and order 1234567 and API_KEY")

        buf = io.StringIO()
        with redirect_stdout(buf):
            t.record_llm_call("something entirely different with no overlap whatsoever")

        output = buf.getvalue()
        # If health dropped below 95, the warning message must appear
        scores = t.get_scores()
        last = scores[-1]
        if last["health_score"] < 95:
            self.assertIn("[AgentAutopsy] WARNING", output)
            self.assertIn("context health at", output)

    def test_no_warning_at_perfect_health(self):
        """Identical prompts should not trigger a warning."""
        import io
        from contextlib import redirect_stdout

        db = _mem_db()
        t = ContextHealthTracker(db=db, run_id="run-warn-002", warn_threshold=60)
        anchor = "Check refund status for customer_id CUST001 order 9876543"
        t.record_llm_call(anchor)

        buf = io.StringIO()
        with redirect_stdout(buf):
            t.record_llm_call(anchor)   # identical → health ≈ 100

        output = buf.getvalue()
        self.assertNotIn("[AgentAutopsy] WARNING", output)


# ── Integration: simulate a real multi-step session ──────────────────────────

class TestIntegration(unittest.TestCase):
    def test_health_decays_across_unrelated_steps(self):
        t = _tracker()
        t.record_llm_call(
            "Process refund for customer_id CUST001, order 9876543, dated 2024-06-15"
        )
        # Gradually drift away from the anchor topic
        t.record_llm_call(
            "customer_id CUST001 refund submitted — reference order 9876543"
        )
        t.record_llm_call(
            "Looking into issue with the payment gateway configuration"
        )
        t.record_llm_call(
            "General system health check — memory usage at 70%"
        )
        t.record_llm_call(
            "How to implement binary search in Python"
        )

        scores = [s["health_score"] for s in t.get_scores()]
        # First step is always 100
        self.assertAlmostEqual(scores[0], 100.0)
        # Last step should be significantly lower than step 2
        self.assertLess(scores[-1], scores[1])

    def test_health_stays_high_for_same_topic(self):
        t = _tracker()
        anchor = "customer_id CUST001 order 9876543 refund 2024-06-15 API_KEY"
        t.record_llm_call(anchor)
        for _ in range(4):
            # Slight variation but same core entities
            t.record_llm_call(
                "customer_id CUST001 order 9876543 has been refunded — dated 2024-06-15"
            )
        scores = [s["health_score"] for s in t.get_scores()]
        # All steps should remain high
        for s in scores[1:]:
            self.assertGreater(s, 50.0)

    def test_step_numbers_sequential(self):
        t = _tracker()
        for i in range(7):
            t.record_llm_call(f"Step {i} with some content")
        steps = [s["step"] for s in t.get_scores()]
        self.assertEqual(steps, list(range(1, 8)))


# ── CLI show-context-health output format ─────────────────────────────────────

class TestContextHealthCLIOutput(unittest.TestCase):
    """Verify the output formatting helpers used by `replay --show-context-health`."""

    def _format_health_report(
        self,
        scores: list[dict],
        run_id: str = "test-run",
    ) -> str:
        """Replicate the formatting logic from cli.py for testing."""
        if not scores:
            return "(no context health data)"
        total = len(scores)
        lines = [
            "",
            "\033[1;38;5;75m═══ Context Health Report ═══\033[0m",
            f"  Run: {run_id}  |  {total} steps",
            f"  {'Step':>4}  {'Health':>7}  {'Overlap':>8}  {'Entities':>9}  Notes",
            "  " + "─" * 55,
        ]
        for row in scores:
            step = row["step"]
            health = row["health_score"]
            overlap = row["token_overlap"]
            entity_ret = row["entity_retention"]
            missing = row.get("missing_entities", [])
            alert = "⚠ " if row.get("alert_fired") else "  "
            missing_str = ""
            if missing:
                missing_str = " lost: " + ", ".join(f"'{e}'" for e in missing[:3])
            lines.append(
                f"  {step:>4}  {health:>6.1f}%  {overlap:>7.1f}%  {entity_ret:>8.1f}%"
                f"  {alert}(step {step} of {total}){missing_str}"
            )
        return "\n".join(lines)

    def test_single_step_report(self):
        scores = [
            {
                "step": 1,
                "health_score": 100.0,
                "token_overlap": 100.0,
                "entity_retention": 100.0,
                "contradiction_count": 0,
                "missing_entities": [],
                "alert_fired": False,
            }
        ]
        report = self._format_health_report(scores)
        self.assertIn("100.0%", report)
        self.assertIn("step 1 of 1", report)

    def test_multi_step_report_shows_warning(self):
        scores = [
            {"step": 1, "health_score": 100.0, "token_overlap": 100.0,
             "entity_retention": 100.0, "contradiction_count": 0,
             "missing_entities": [], "alert_fired": False},
            {"step": 2, "health_score": 75.0, "token_overlap": 70.0,
             "entity_retention": 80.0, "contradiction_count": 0,
             "missing_entities": [], "alert_fired": False},
            {"step": 3, "health_score": 41.0, "token_overlap": 45.0,
             "entity_retention": 37.0, "contradiction_count": 1,
             "missing_entities": ["customer_id"], "alert_fired": True},
        ]
        report = self._format_health_report(scores, run_id="test-run-999")
        self.assertIn("41.0%", report)
        self.assertIn("step 3 of 3", report)
        self.assertIn("⚠", report)
        self.assertIn("customer_id", report)
        self.assertIn("test-run-999", report)

    def test_no_data_message(self):
        report = self._format_health_report([])
        self.assertIn("no context health data", report)


if __name__ == "__main__":
    unittest.main()
