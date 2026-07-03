#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from foundry.causal import analyze_pairs
from foundry.common import FoundryError, dump_json, load_json
from foundry.fuzzing import run_fuzzer
from foundry.genome import propose_mutations
from foundry.guardrails import check_registry, promote_failure
from foundry.portfolio import build_portfolio
from foundry.replay import score_replay
from foundry.research import (
    add_hypothesis,
    complete_experiment,
    init_state,
    next_hypothesis,
    start_experiment,
    status,
)


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _objective(value: str) -> dict[str, str]:
    try:
        name, direction = value.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("objective must use NAME:min or NAME:max") from exc
    if not name or direction not in {"min", "max"}:
        raise argparse.ArgumentTypeError("objective must use NAME:min or NAME:max")
    return {"name": name, "direction": direction}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executable kernel engineering and learning primitives.")
    capabilities = parser.add_subparsers(dest="capability", required=True)

    guardrail = capabilities.add_parser("guardrail", help="Compile and execute reusable failure guardrails.")
    guardrail_commands = guardrail.add_subparsers(dest="command", required=True)
    promote = guardrail_commands.add_parser("promote")
    promote.add_argument("--failure", required=True, type=Path)
    promote.add_argument("--registry", required=True, type=Path)
    check = guardrail_commands.add_parser("check")
    check.add_argument("--registry", required=True, type=Path)
    check.add_argument("--facts", required=True, type=Path)
    check.add_argument("--scope", action="append", default=[])
    check.add_argument("--out", type=Path)

    fuzz = capabilities.add_parser("fuzz", help="Run deterministic semantic counterexample search.")
    fuzz_commands = fuzz.add_subparsers(dest="command", required=True)
    fuzz_run = fuzz_commands.add_parser("run")
    fuzz_run.add_argument("--adapter", required=True, type=Path)
    fuzz_run.add_argument("--iterations", type=int, default=100)
    fuzz_run.add_argument("--seed", type=int, default=0)
    fuzz_run.add_argument("--out", required=True, type=Path)
    fuzz_run.add_argument("--stop-after", type=int, default=0)

    research = capabilities.add_parser("research", help="Manage a bounded autonomous experiment state machine.")
    research_commands = research.add_subparsers(dest="command", required=True)
    research_init = research_commands.add_parser("init")
    research_init.add_argument("--state", required=True, type=Path)
    research_init.add_argument("--project", required=True)
    research_init.add_argument("--mode", choices=("quick", "standard", "research"), default="research")
    research_init.add_argument("--contract", required=True)
    research_init.add_argument("--objective", action="append", type=_objective, required=True)
    research_init.add_argument("--max-experiments", type=int, default=10)
    research_init.add_argument("--max-tpu-hours", type=float, default=4.0)
    research_init.add_argument("--max-failures", type=int, default=3)
    research_add = research_commands.add_parser("add")
    research_add.add_argument("--state", required=True, type=Path)
    research_add.add_argument("--hypothesis", required=True, type=Path)
    research_next = research_commands.add_parser("next")
    research_next.add_argument("--state", required=True, type=Path)
    research_start = research_commands.add_parser("start")
    research_start.add_argument("--state", required=True, type=Path)
    research_start.add_argument("--hypothesis-id", required=True)
    research_start.add_argument("--experiment-id", required=True)
    research_complete = research_commands.add_parser("complete")
    research_complete.add_argument("--state", required=True, type=Path)
    research_complete.add_argument("--experiment-id", required=True)
    research_complete.add_argument("--result", required=True, type=Path)
    research_status = research_commands.add_parser("status")
    research_status.add_argument("--state", required=True, type=Path)

    replay = capabilities.add_parser("replay", help="Score historical task replays across skill variants.")
    replay_commands = replay.add_subparsers(dest="command", required=True)
    replay_score = replay_commands.add_parser("score")
    replay_score.add_argument("--suite", required=True, type=Path)
    replay_score.add_argument("--results", required=True, type=Path)
    replay_score.add_argument("--out", required=True, type=Path)

    portfolio = capabilities.add_parser("portfolio", help="Build exact evidence-backed dispatch rules.")
    portfolio_commands = portfolio.add_subparsers(dest="command", required=True)
    portfolio_build = portfolio_commands.add_parser("build")
    portfolio_build.add_argument("--input", required=True, type=Path)
    portfolio_build.add_argument("--metric", required=True)
    portfolio_build.add_argument("--direction", choices=("min", "max"), default="min")
    portfolio_build.add_argument("--min-repeats", type=int, default=1)
    portfolio_build.add_argument("--min-relative-margin", type=float, default=0.0)
    portfolio_build.add_argument("--out", required=True, type=Path)

    causal = capabilities.add_parser("causal", help="Analyze controlled source/HLO observation pairs.")
    causal_commands = causal.add_subparsers(dest="command", required=True)
    causal_analyze = causal_commands.add_parser("analyze")
    causal_analyze.add_argument("--input", required=True, type=Path)
    causal_analyze.add_argument("--metric", required=True)
    causal_analyze.add_argument("--out", required=True, type=Path)

    genome = capabilities.add_parser("genome", help="Propose one-gene traceable kernel mutations.")
    genome_commands = genome.add_subparsers(dest="command", required=True)
    genome_propose = genome_commands.add_parser("propose")
    genome_propose.add_argument("--spec", required=True, type=Path)
    genome_propose.add_argument("--out", required=True, type=Path)
    genome_propose.add_argument("--limit", type=int, default=20)
    return parser


def dispatch(args: argparse.Namespace) -> tuple[Any, int]:
    if args.capability == "guardrail" and args.command == "promote":
        return promote_failure(args.failure, args.registry), 0
    if args.capability == "guardrail" and args.command == "check":
        report = check_registry(args.registry, args.facts, args.scope)
        if args.out:
            dump_json(args.out, report)
        return report, 0 if report["passed"] else 2
    if args.capability == "fuzz" and args.command == "run":
        report = run_fuzzer(args.adapter, args.iterations, args.seed, args.out, args.stop_after)
        return report, 0 if report["passed"] else 2
    if args.capability == "research" and args.command == "init":
        return init_state(
            args.state,
            args.project,
            args.mode,
            args.contract,
            args.objective,
            args.max_experiments,
            args.max_tpu_hours,
            args.max_failures,
        ), 0
    if args.capability == "research" and args.command == "add":
        return add_hypothesis(args.state, load_json(args.hypothesis)), 0
    if args.capability == "research" and args.command == "next":
        return {"next_hypothesis": next_hypothesis(args.state)}, 0
    if args.capability == "research" and args.command == "start":
        return start_experiment(args.state, args.hypothesis_id, args.experiment_id), 0
    if args.capability == "research" and args.command == "complete":
        return complete_experiment(args.state, args.experiment_id, load_json(args.result)), 0
    if args.capability == "research" and args.command == "status":
        return status(args.state), 0
    if args.capability == "replay" and args.command == "score":
        return score_replay(args.suite, args.results, args.out), 0
    if args.capability == "portfolio" and args.command == "build":
        return build_portfolio(
            args.input,
            args.metric,
            args.direction,
            args.out,
            args.min_repeats,
            args.min_relative_margin,
        ), 0
    if args.capability == "causal" and args.command == "analyze":
        return analyze_pairs(args.input, args.metric, args.out), 0
    if args.capability == "genome" and args.command == "propose":
        return propose_mutations(args.spec, args.out, args.limit), 0
    raise FoundryError(f"Unsupported command: {args.capability} {args.command}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload, code = dispatch(args)
        _print(payload)
        return code
    except FoundryError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
