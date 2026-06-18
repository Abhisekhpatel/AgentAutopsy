"""AgentAutopsy — when your agent fails, this tells you exactly why."""

from __future__ import annotations

import atexit

from agentautopsy.db import create_tables, get_db, insert_run
from agentautopsy.interceptor import (
    start_anthropic_interceptor,
    start_http_interceptor,
    start_interceptor,
)
from agentautopsy.mcp_handler import MCPAutopsy
from agentautopsy.reporter import print_report
from agentautopsy.dvr_replay import DVRReplay
from agentautopsy.eval_generator import EvalGenerator
from agentautopsy.loop_detector import LoopDetector
from agentautopsy.schema_drift import SchemaDriftDetector
from agentautopsy.context_monitor import ContextMonitor
from agentautopsy.context_health import ContextHealthTracker

__all__ = [
    "ContextHealthTracker",
    "ContextMonitor",
    "DVRReplay",
    "EvalGenerator",
    "LoopDetector",
    "MCPAutopsy",
    "SchemaDriftDetector",
    "generate_fingerprint",
    "get_callback_handler",
    "get_crewai_handler",
    "get_langgraph_handler",
    "list_fingerprints",
    "watch",
    "watch_mcp",
]

_watch_context: tuple[str, object] | None = None
_mcp_autopsy: object | None = None

def get_callback_handler():
    """Return a LangChain callback handler for the active watch() run."""
    if _watch_context is None:
        raise RuntimeError("Call agentautopsy.watch() before get_callback_handler()")
    run_id, db = _watch_context
    from agentautopsy.langchain_handler import AgentAutopsyCallbackHandler

    return AgentAutopsyCallbackHandler(run_id, db)

def get_langgraph_handler():
    """Return a LangGraph callback handler for the active watch() run."""
    if _watch_context is None:
        raise RuntimeError("Call agentautopsy.watch() before get_langgraph_handler()")
    run_id, db = _watch_context
    from agentautopsy.langgraph_handler import AgentAutopsyLangGraphHandler

    return AgentAutopsyLangGraphHandler(run_id, db)

def get_crewai_handler():
    """Return a CrewAI callback handler for the active watch() run."""
    if _watch_context is None:
        raise RuntimeError("Call agentautopsy.watch() before get_crewai_handler()")
    run_id, db = _watch_context
    from agentautopsy.crewai_handler import AgentAutopsyCrewAIHandler

    return AgentAutopsyCrewAIHandler(run_id, db)

def generate_fingerprint(trace: dict) -> str:  # type: ignore[type-arg]
    """Generate a failure fingerprint for a trace dict. See fingerprint.py."""
    from agentautopsy.fingerprint import generate_fingerprint as _gf

    return _gf(trace)

def list_fingerprints() -> list:  # type: ignore[type-arg]
    """Return all stored failure fingerprints sorted by occurrence count."""
    from agentautopsy.fingerprint import list_fingerprints as _lf

    db = get_db()
    create_tables(db)
    return _lf(db)

def watch_mcp(
    server_name: str | None = None,
    *,
    agent_name: str | None = None,
    parent_run_id: str | None = None,
):
    """Start MCP post-mortem tracing — one import, one line."""
    global _mcp_autopsy

    _mcp_autopsy = MCPAutopsy.start(
        server_name=server_name,
        agent_name=agent_name,
        parent_run_id=parent_run_id,
    )
    return _mcp_autopsy

def watch(
    agent_name: str | None = None,
    parent_run_id: str | None = None,
):
    global _watch_context

    db = get_db()
    create_tables(db)
    from agentautopsy.cache import setup_cache

    setup_cache(db)
    run_id = insert_run(
        db,
        agent_name=agent_name,
        parent_run_id=parent_run_id,
    )
    _watch_context = (run_id, db)
    start_interceptor(run_id, db)
    start_anthropic_interceptor(run_id, db)
    start_http_interceptor(run_id, db)
    SchemaDriftDetector(run_id=run_id, db=db, agent_name=agent_name or "agent").watch()
    DVRReplay(db=db, run_id=run_id).watch()
    EvalGenerator(db=db, run_id=run_id, agent_name=agent_name or "agent").watch()
    LoopDetector(db=db, run_id=run_id, agent_name=agent_name or "agent").watch()
    ContextMonitor(db=db, run_id=run_id, agent_name=agent_name or "agent").watch()
    # Context Decay Detector — wraps interceptors last so it fires after all other hooks
    ContextHealthTracker(db=db, run_id=run_id, agent_name=agent_name or "agent").watch()

    import sys

    _original_excepthook = sys.excepthook

    def _autopsy_excepthook(exc_type, exc_value, exc_traceback):
        from agentautopsy.db import insert_event
        from agentautopsy.loop_detector import LoopKillException, record_call_event

        # Feed the exception itself as an error event so LoopDetector sees it
        record_call_event(
            "error",
            {"error_type": exc_type.__name__, "message": str(exc_value)},
        )
        insert_event(
            db,
            run_id,
            "error",
            {"error_type": exc_type.__name__, "message": str(exc_value)},
        )
        if not issubclass(exc_type, LoopKillException):
            _original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _autopsy_excepthook
    import time

    label = agent_name or "agent"
    print("\n\033[38;5;39m" + "━" * 60 + "\033[0m")
    time.sleep(0.1)
    print("\033[1;38;5;82m⚡ [AgentAutopsy] Engine Initialized\033[0m")
    time.sleep(0.1)
    print(f"\033[38;5;244m▶ Target: \033[1;37m{label}\033[0m")
    time.sleep(0.1)
    print(f"\033[38;5;244m▶ Session: \033[38;5;141m{run_id}\033[0m")
    time.sleep(0.1)
    if parent_run_id:
        print(f"\033[38;5;244m▶ Parent: \033[38;5;141m{parent_run_id}\033[0m")
        time.sleep(0.1)
    print(
        "\033[38;5;244m▶ Status: \033[38;5;11mIntercepting LLM & HTTP Traffic in real-time...\033[0m"
    )
    time.sleep(0.1)
    print("\033[38;5;39m" + "━" * 60 + "\033[0m\n")

    def on_exit():
        from agentautopsy.analyzer import _parse_analysis, analyze
        from agentautopsy.cache import lookup_fix, store_fix
        from agentautopsy.detector import detect_failure, take_snapshot
        from agentautopsy.pruner import prune
        from agentautopsy.replay import replay

        result = detect_failure(run_id, db)
        if not result["failed"]:
            from agentautopsy.db import mark_run_completed
            from agentautopsy.loop_detector import get_active_detector

            mark_run_completed(db, run_id)
            loop_det = get_active_detector()
            loop_stats = loop_det.current_stats() if loop_det else {}
            from agentautopsy.context_monitor import get_active_monitor

            ctx_mon = get_active_monitor()
            ctx_pct = ctx_mon.current_pct() if ctx_mon else 0.0

            from agentautopsy.context_health import get_active_tracker

            ctx_health = get_active_tracker()
            ctx_health_score = ctx_health.current_score() if ctx_health else 100.0

            print("\n\033[38;5;39m" + "━" * 60 + "\033[0m")
            time.sleep(0.1)
            print("\033[1;38;5;82m✅ [AgentAutopsy] Analysis Complete\033[0m")
            time.sleep(0.1)
            print(
                f"\033[38;5;244m▶ Run \033[38;5;141m{run_id}\033[38;5;244m executed flawlessly.\033[0m"
            )
            time.sleep(0.1)
            if loop_stats:
                cost = loop_stats.get("total_cost_usd", 0)
                tokens = loop_stats.get("total_tokens", 0)
                print(
                    f"\033[38;5;244m▶ Cost: \033[38;5;82m${cost:.4f}\033[38;5;244m Tokens: {tokens}\033[0m"
                )
                time.sleep(0.1)
            if ctx_pct:
                ctx_color = "82" if ctx_pct < 70 else ("226" if ctx_pct < 90 else "196")
                print(
                    f"\033[38;5;244m▶ Context: \033[38;5;{ctx_color}m{ctx_pct:.1f}% of window used\033[0m"
                )
                time.sleep(0.1)
            if ctx_health_score < 100.0:
                health_color = (
                    "82" if ctx_health_score >= 80
                    else "226" if ctx_health_score >= 60
                    else "208"
                )
                print(
                    f"\033[38;5;244m▶ Context health: \033[38;5;{health_color}m{ctx_health_score:.0f}%\033[0m"
                )
                time.sleep(0.1)
            print(
                "\033[38;5;244m▶ Type \033[1;37magentautopsy ui\033[38;5;244m in your terminal to view the trace graph.\033[0m"
            )
            time.sleep(0.1)
            print("\033[38;5;39m" + "━" * 60 + "\033[0m\n")
            return

        from agentautopsy.db import mark_run_failed

        mark_run_failed(db, run_id)

        from agentautopsy.eval_generator import generate_eval_for_run

        eval_path = generate_eval_for_run(run_id, db)

        time.sleep(0.1)
        print("\n\033[1;38;5;196m❌ [AgentAutopsy] Critical Failure Intercepted\033[0m")
        time.sleep(0.1)
        print(f"\033[38;5;244m▶ Error: \033[1;38;5;196m{result['error_type']}\033[0m")
        time.sleep(0.1)
        print(f"\033[38;5;244m▶ Trace: \033[38;5;196m{result['message']}\033[0m")
        if eval_path:
            time.sleep(0.1)
            print(
                f"\033[38;5;244m▶ Eval: \033[1;38;5;82mRegression test generated → {eval_path}\033[0m"
            )

        # Build the snapshot now — needed for both fingerprinting and analysis.
        snapshot = take_snapshot(run_id, db)
        pruned = prune(snapshot, result.get("failure_event_id"))

        # ------------------------------------------------------------------
        # Fingerprint check — instant lookup, zero AI calls
        # ------------------------------------------------------------------
        from agentautopsy.fingerprint import (
            ensure_fingerprint_tables,
            generate_fingerprint as _gen_fp,
            match_fingerprint,
            print_fingerprint_match,
            record_fingerprint,
            update_fingerprint_fix,
        )
        ensure_fingerprint_tables(db)
        _fp_trace = {**result, "events": pruned}
        _fp_id = _gen_fp(_fp_trace)
        _fp_match = match_fingerprint(db, _fp_id)

        if _fp_match and _fp_match.get("example_fix"):
            # Known pattern with a stored fix — surface it immediately.
            record_fingerprint(db, _fp_id, _fp_trace)
            print_fingerprint_match(_fp_id, _fp_match)
            print_report(run_id, db)
            return

        # ------------------------------------------------------------------
        # Cache check (error-type + message hash, no AI)
        # ------------------------------------------------------------------
        cached = lookup_fix(db, result["error_type"], result["message"])
        if cached:
            time.sleep(0.8)
            print(
                "\n\033[38;5;39m▶ \033[1;38;5;141mAI Root Cause Analysis Triggered...\033[0m"
            )
            time.sleep(0.8)
            print(
                "\033[38;5;39m▶ \033[1;38;5;82mCache Hit — Fix Found Instantly\033[0m\n"
            )
            time.sleep(0.1)
            print(cached)
            record_fingerprint(db, _fp_id, _fp_trace)
            return

        # ------------------------------------------------------------------
        # Full AI analysis
        # ------------------------------------------------------------------
        _fp_fix: str | None = None
        try:
            analysis = analyze(pruned, result)
            print(f"\n[AgentAutopsy] analysis:\n{analysis}")

            replay_result = replay(run_id, db, analysis)
            if replay_result["verified"]:
                print("\n[AgentAutopsy] fix verified ✓")
                print("✓ Replay passed")
                print("✓ Failure resolved")
                store_fix(
                    db, result["error_type"], result["message"], analysis, verified=True
                )
                # Extract the structured FIX line to store in the fingerprint
                _, _fp_fix = _parse_analysis(analysis)
                if _fp_fix:
                    update_fingerprint_fix(db, _fp_id, _fp_fix)
            else:
                print("\n[AgentAutopsy] fix not verified — review manually")
        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                print(
                    "\n[AgentAutopsy] Auto-fix bypassed: LLM authentication failed (check ANTHROPIC_API_KEY)."
                )
            else:
                print(f"\n[AgentAutopsy] Auto-fix failed: {e}")
        finally:
            # Always record this occurrence so the pattern accumulates over time.
            record_fingerprint(db, _fp_id, _fp_trace, fix=_fp_fix)

        print_report(run_id, db)

    atexit.register(on_exit)
