"""Command-line interface for AgentAutopsy."""

import json
import sys

from agentautopsy.cache import cache_stats, setup_cache
from agentautopsy.db import create_tables, get_db
from agentautopsy.reporter import print_report

def _usage() -> None:
    print(
        """Usage: agentautopsy <command>

Commands:
  runs                    List all runs (id, start_time, status)
  agents                  List multi-agent run chains
  replay <run_id>         Print the event report for a run
  replay --run-id <id> --from-step <n>  DVR replay from a step
  share <run_id>          Export a run trace to a shareable JSON file
  fix <run_id>            Apply an automated fix for a failed run
  generate-evals          Generate pytest tests from all recorded failures
  loops                   Show all detected loops from recorded runs
  context                 Show context window usage across all runs
  patterns                List known failure patterns sorted by frequency
  stats                   Show fix cache statistics
  serve                   Start HTTP API for Monadix (POST /analyze)
  ui                      Open the web UI in your browser
  mcp <cmd...>            Run an MCP server and proxy stdio to trace it

Examples:
  agentautopsy runs
  agentautopsy agents
  agentautopsy replay abc-123-def
  agentautopsy share abc-123-def
  agentautopsy fix abc-123-def
  agentautopsy fix abc-123-def --create-pr
  agentautopsy generate-evals
  agentautopsy generate-evals --run-id abc-123-def
  agentautopsy loops
  agentautopsy loops --run-id abc-123-def
  agentautopsy context
  agentautopsy context --run-id abc-123-def
  agentautopsy patterns
  agentautopsy prune [days]
  agentautopsy stats
  agentautopsy serve
  agentautopsy ui"""
    )

def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        _usage()
        return

    cmd = argv[0]
    db = get_db()
    create_tables(db)

    if cmd == "prune":
        from agentautopsy.db import prune_old_runs

        days = 7
        if len(argv) > 1:
            try:
                days = int(argv[1])
            except ValueError:
                print("Usage: agentautopsy prune [days]")
                return
        count = prune_old_runs(db, days=days)
        print(
            f"Pruned {count} runs older than {days} days. Database compressed via VACUUM."
        )
        return

    if cmd == "runs":
        if not db["runs"].exists():
            print("No runs table yet.")
            return
        rows = list(db["runs"].rows_where(order_by="start_time desc"))
        if not rows:
            print("No runs found.")
            return
        for row in rows:
            agent = row.get("agent_name") or "agent"
            parent = row.get("parent_run_id") or ""
            parent_suffix = f"\tparent={parent[:8]}..." if parent else ""
            print(
                f"{row['id']}\t{agent}\t{row['start_time']}\t{row['status']}{parent_suffix}"
            )
        return

    if cmd == "agents":
        from agentautopsy.ui import _load_data, build_agent_chains

        if not db["runs"].exists():
            print("No runs table yet.")
            return
        runs, runs_data = _load_data(db)
        chains = build_agent_chains(runs, runs_data)
        if not chains:
            print("No agent runs found.")
            return
        for index, chain in enumerate(chains, start=1):
            labels = " → ".join(
                f"{node['agent_name']} [{node['status']}]" for node in chain["nodes"]
            )
            tokens = sum(node.get("total_tokens", 0) for node in chain["nodes"])
            print(f"Chain {index} (root: {chain['root_id'][:8]}...)")
            print(f"  {labels}")
            print(f"  total_tokens: {tokens}")
            print()
        return

    if cmd == "replay":
        if "--run-id" in argv or "--from-step" in argv:
            from agentautopsy.dvr_replay import DVRReplay

            run_id = None
            from_step = 1
            if "--run-id" in argv:
                run_id = argv[argv.index("--run-id") + 1]
            if "--from-step" in argv:
                from_step = int(argv[argv.index("--from-step") + 1])
            if not run_id:
                print(
                    "usage: agentautopsy replay --run-id <id> [--from-step <n>]",
                    file=sys.stderr,
                )
                sys.exit(2)
            dvr = DVRReplay(db=db)
            try:
                result = dvr.replay_from_step(run_id, from_step)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                sys.exit(1)
            print(json.dumps(result, indent=2, default=str))
            return

        if len(argv) < 2:
            print("usage: agentautopsy replay <run_id>", file=sys.stderr)
            sys.exit(2)
        run_id = argv[1]
        print_report(run_id, db)
        return

    if cmd == "share":
        if len(argv) < 2:
            print("usage: agentautopsy share <run_id>", file=sys.stderr)
            sys.exit(2)
        run_id = argv[1]
        from agentautopsy.share import share_run

        try:
            path = share_run(run_id)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(path)
        return

    if cmd == "fix":
        if len(argv) < 2:
            print("usage: agentautopsy fix <run_id> [--create-pr]", file=sys.stderr)
            sys.exit(2)
        run_id = argv[1]
        create_pr_flag = "--create-pr" in argv[2:]
        from agentautopsy.autofix import apply_fix
        from agentautopsy.github_pr import create_pr

        result = apply_fix(run_id)
        print(json.dumps(result, indent=2))
        if create_pr_flag:
            if not result.get("success"):
                print(
                    "Skipping PR creation because fix did not succeed.", file=sys.stderr
                )
                sys.exit(1)
            try:
                pr = create_pr(
                    run_id,
                    result.get("fix", ""),
                    error_type=str(result.get("error_type") or "agent failure"),
                    root_cause=str(result.get("root_cause") or ""),
                    fix_applied=str(result.get("fix") or ""),
                    test_results=str(result.get("test_output") or ""),
                    file_path=result.get("file_path"),
                )
                print(json.dumps(pr, indent=2))
            except Exception as exc:
                print(f"PR creation failed: {exc}", file=sys.stderr)
                sys.exit(1)
        sys.exit(0 if result.get("success") else 1)

    if cmd == "generate-evals":
        from agentautopsy.eval_generator import EvalGenerator

        generator = EvalGenerator(db=db)
        if "--run-id" in argv:
            run_id = argv[argv.index("--run-id") + 1]
            path = generator.generate_from_run(run_id)
            if path:
                print(f"Generated regression test: {path}")
            else:
                print(f"No failure found for run {run_id}; nothing generated.")
            return
        paths = generator.generate_all()
        if not paths:
            print("No recorded failures found. Nothing to generate.")
            return
        print(f"Generated {len(paths)} regression test(s):")
        for path in paths:
            print(f"  {path}")
        return

    if cmd == "loops":
        from agentautopsy.loop_detector import cost_per_run, load_loop_events

        filter_run = None
        if "--run-id" in argv:
            filter_run = argv[argv.index("--run-id") + 1]

        events = load_loop_events(db)
        if filter_run:
            events = [e for e in events if e["run_id"] == filter_run]

        if not events:
            print("No loop events recorded.")
        else:
            print(f"{'TYPE':<22} {'RUN':<14} {'STEP':>5} {'TOKENS':>8} {'COST':>9} LABEL")
            print("─" * 90)
            for ev in events:
                rid = ev["run_id"][:12] + "..."
                killed_marker = " [killed]" if ev["killed"] else ""
                print(
                    f"{ev['loop_type']:<22} {rid:<14} {ev['trigger_step']:>5} "
                    f"{ev['total_tokens']:>8} ${ev['total_cost_usd']:>7.4f} "
                    f"{ev['trigger_label']}{killed_marker}"
                )

        print()
        runs_cost = cost_per_run(db)
        if runs_cost:
            print(f"{'RUN':<14} {'AGENT':<18} {'STATUS':<12} {'TOKENS':>8} {'COST':>9}")
            print("─" * 72)
            for rc in runs_cost:
                rid = rc["run_id"][:12] + "..."
                print(
                    f"{rid:<14} {rc['agent_name']:<18} {rc['status']:<12} "
                    f"{rc['total_tokens']:>8} ${rc['cost_usd']:>7.4f}"
                )
        return

    if cmd == "context":
        from agentautopsy.context_monitor import load_context_snapshots, load_context_ui_data

        run_id_arg = None
        if "--run-id" in argv:
            run_id_arg = argv[argv.index("--run-id") + 1]

        if run_id_arg:
            snaps = load_context_snapshots(db, run_id_arg)
            if not snaps:
                print(f"No context snapshots found for run {run_id_arg}.")
                return
            print(f"\nContext window usage for run {run_id_arg}")
            print(f"{'STEP':>5} {'MODEL':<20} {'TOKENS':>8} {'LIMIT':>8} {'%':>7} LEVEL")
            print("─" * 72)
            for s in snaps:
                level_color = (
                    "\033[38;5;196m" if s["alert_level"] == "critical"
                    else "\033[38;5;226m" if s["alert_level"] == "warn"
                    else "\033[38;5;82m"
                )
                print(
                    f"{s['step']:>5} {s['model']:<20} {s['tokens_used']:>8} "
                    f"{s['context_limit']:>8} {s['pct_used']:>6.1f}% "
                    f"{level_color}{s['alert_level']}\033[0m"
                )
                if s.get("truncation_suspected"):
                    print(f"  \033[38;5;208m⚠ Silent truncation suspected at step {s['step']}\033[0m")
                for sug in s.get("suggestions") or []:
                    print(f"  \033[38;5;244m→ {sug}\033[0m")
        else:
            ui_data = load_context_ui_data(db)
            by_run = ui_data.get("by_run", {})
            if not by_run:
                print("No context data recorded yet. Run agentautopsy.watch() first.")
                return
            print(f"\n{'RUN':<14} {'AGENT':<18} {'MODEL':<18} {'TOKENS':>8} {'LIMIT':>8} {'%':>7} ALERT")
            print("─" * 90)
            for rid, info in by_run.items():
                snaps = info.get("steps") or []
                if not snaps:
                    continue
                latest = snaps[-1]
                level_color = (
                    "\033[38;5;196m" if info["alert_level"] == "critical"
                    else "\033[38;5;226m" if info["alert_level"] == "warn"
                    else "\033[38;5;82m"
                )
                short_rid = rid[:12] + "..."
                print(
                    f"{short_rid:<14} {info['agent_name']:<18} {info['model']:<18} "
                    f"{latest['tokens_used']:>8} {info['context_limit']:>8} "
                    f"{info['latest_pct']:>6.1f}% "
                    f"{level_color}{info['alert_level']}\033[0m"
                )
        return

    if cmd == "patterns":
        from agentautopsy.fingerprint import ensure_fingerprint_tables, list_fingerprints

        ensure_fingerprint_tables(db)
        records = list_fingerprints(db)
        if not records:
            print("No failure patterns recorded yet.")
            return
        print(
            f"\n{'FINGERPRINT':<12} {'OCCURRENCES':>11} {'CONFIDENCE':<12} "
            f"{'ERROR TYPE':<25} {'TOOL':<20} {'POSITION':<8} FIX"
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
        return

    if cmd == "stats":
        setup_cache(db)
        stats = cache_stats(db)
        print(f"total_fixes: {stats['total_fixes']}")
        print(f"total_hits: {stats['total_hits']}")

        if db["events"].exists():
            row = db.execute(
                """
                SELECT
                    COALESCE(SUM(token_input), 0),
                    COALESCE(SUM(token_output), 0),
                    COALESCE(SUM(cost_usd), 0.0)
                FROM events
                """
            ).fetchone()
            token_input = int(row[0]) if row else 0
            token_output = int(row[1]) if row else 0
            total_cost = float(row[2]) if row else 0.0
            print(f"total_tokens_input: {token_input}")
            print(f"total_tokens_output: {token_output}")
            print(f"total_tokens: {token_input + token_output}")
            print(f"total_cost_usd: {total_cost:.6f}")
        else:
            print("total_tokens_input: 0")
            print("total_tokens_output: 0")
            print("total_tokens: 0")
            print("total_cost_usd: 0.000000")
        return

    if cmd == "serve":
        from agentautopsy.api import start_api_server

        host = None
        port = None
        if "--host" in argv:
            host = argv[argv.index("--host") + 1]
        if "--port" in argv:
            port = int(argv[argv.index("--port") + 1])
        try:
            start_api_server(host=host, port=port)
        except KeyboardInterrupt:
            print("\nAPI stopped.")
        return

    if cmd == "ui":
        from agentautopsy.ui import start_ui

        try:
            start_ui()
        except KeyboardInterrupt:
            print("\nUI stopped.")
        return

    if cmd == "mcp":
        if len(argv) < 2:
            print("usage: agentautopsy mcp [args...]", file=sys.stderr)
            sys.exit(2)
        mcp_cmd = argv[1:]
        from agentautopsy.mcp_interceptor import run_mcp_proxy

        run_mcp_proxy(mcp_cmd)
        return

    print(f"Unknown command: {cmd}", file=sys.stderr)
    _usage()
    sys.exit(2)
