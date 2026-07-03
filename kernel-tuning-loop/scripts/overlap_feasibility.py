#!/usr/bin/env python3
"""Evaluate communication-compute overlap from measured pipeline probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DECISIONS = {
  "proceed_to_pipeline_design",
  "run_structural_breakdown",
  "change_chunk_or_tile_size",
  "optimize_state_lifetime",
  "reject_expression_order_tuning",
  "keep_current_baseline",
}


def _speedup(baseline: float | None, candidate: float | None) -> float | None:
  if baseline is None or candidate is None or baseline <= 0:
    return None
  return (baseline - candidate) / baseline


def _decision(args: argparse.Namespace, metrics: dict) -> tuple[str, str]:
  tol = args.tolerance
  if metrics["memory_fit"] is False:
    return "change_chunk_or_tile_size", "双缓冲与状态驻留超过可用片上内存。"
  if (
    args.no_comm_multi_step_ms is not None
    and args.full_baseline_ms is not None
    and args.no_comm_multi_step_ms > args.full_baseline_ms * (1 + tol)
  ):
    return "optimize_state_lifetime", "无通信多步计算已经慢于基线，状态生命周期是先决瓶颈。"
  if args.comm_ms > args.compute_ms * (1 + tol):
    return "change_chunk_or_tile_size", "通信窗口大于可隐藏的计算窗口。"
  if abs(args.candidate_ms - args.serial_ms) <= max(args.serial_ms, 1e-9) * tol:
    return "reject_expression_order_tuning", "候选调度与串行/自然调度等价。"
  full_speedup = metrics["full_speedup"]
  if full_speedup is not None and full_speedup <= tol:
    if metrics["hidden_fraction"] >= 0.5:
      return "run_structural_breakdown", "探针存在 overlap，但 full latency 没有形成可接受收益。"
    return "keep_current_baseline", "探针和完整 kernel 均未证明可接受收益。"
  if metrics["exposed_ms"] <= max(args.compute_ms, args.comm_ms) * tol:
    return "proceed_to_pipeline_design", "候选时间接近 max(C, M)，且可行性门禁通过。"
  return "run_structural_breakdown", "通信部分隐藏，但仍有暴露时间或结构开销需要拆分。"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--compute-ms", type=float, required=True)
  parser.add_argument("--comm-ms", type=float, required=True)
  parser.add_argument("--serial-ms", type=float, required=True)
  parser.add_argument("--candidate-ms", type=float, required=True)
  parser.add_argument("--full-baseline-ms", type=float)
  parser.add_argument("--full-candidate-ms", type=float)
  parser.add_argument("--no-comm-multi-step-ms", type=float)
  parser.add_argument("--resident-bytes", type=int)
  parser.add_argument("--memory-budget-bytes", type=int)
  parser.add_argument("--comm-bytes", type=int)
  parser.add_argument("--achieved-bandwidth-gbps", type=float)
  parser.add_argument("--tolerance", type=float, default=0.05)
  parser.add_argument("--json-out", type=Path)
  parser.add_argument("--markdown-out", type=Path)
  args = parser.parse_args()

  measured = {
    "compute_ms": args.compute_ms,
    "comm_ms": args.comm_ms,
    "serial_ms": args.serial_ms,
    "candidate_ms": args.candidate_ms,
  }
  if any(value < 0 for value in measured.values()):
    parser.error("measured times must be non-negative")
  for name in ("full_baseline_ms", "full_candidate_ms", "no_comm_multi_step_ms"):
    value = getattr(args, name)
    if value is not None and value < 0:
      parser.error(f"{name.replace('_', '-')} must be non-negative")
  if args.tolerance < 0:
    parser.error("tolerance must be non-negative")
  if args.resident_bytes is not None and args.resident_bytes < 0:
    parser.error("resident-bytes must be non-negative")
  if args.memory_budget_bytes is not None and args.memory_budget_bytes < 0:
    parser.error("memory-budget-bytes must be non-negative")

  hidden_ms_raw = args.compute_ms + args.comm_ms - args.candidate_ms
  hidden_capacity = min(args.compute_ms, args.comm_ms)
  hidden_ms = min(max(hidden_ms_raw, 0.0), hidden_capacity)
  hidden_fraction = hidden_ms / hidden_capacity if hidden_capacity > 0 else 0.0
  overlap_residual_ms = args.candidate_ms - max(args.compute_ms, args.comm_ms)
  exposed_ms = max(overlap_residual_ms, 0.0)
  memory_fit = None
  if args.resident_bytes is not None and args.memory_budget_bytes is not None:
    memory_fit = args.resident_bytes <= args.memory_budget_bytes
  estimated_comm_ms = None
  if args.comm_bytes is not None and args.achieved_bandwidth_gbps:
    estimated_comm_ms = args.comm_bytes / (args.achieved_bandwidth_gbps * 1e9) * 1e3

  metrics = {
    "compute_ms": args.compute_ms,
    "comm_ms": args.comm_ms,
    "serial_ms": args.serial_ms,
    "candidate_ms": args.candidate_ms,
    "necessary_condition_comm_le_compute": args.comm_ms <= args.compute_ms,
    "hidden_ms_raw": hidden_ms_raw,
    "hidden_ms": hidden_ms,
    "hidden_fraction": hidden_fraction,
    "overlap_residual_ms": overlap_residual_ms,
    "exposed_ms": exposed_ms,
    "candidate_vs_serial_speedup": _speedup(args.serial_ms, args.candidate_ms),
    "full_speedup": _speedup(args.full_baseline_ms, args.full_candidate_ms),
    "memory_fit": memory_fit,
    "estimated_comm_ms": estimated_comm_ms,
  }
  decision, reason = _decision(args, metrics)
  assert decision in DECISIONS
  payload = {"metrics": metrics, "decision": decision, "reason": reason}
  rendered = json.dumps(payload, indent=2, ensure_ascii=False)
  print(rendered)

  if args.json_out:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(rendered + "\n", encoding="utf-8")
  if args.markdown_out:
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
      "# Overlap 可行性报告",
      "",
      "## 测量值",
      "",
      "| C | M | S | O | hidden fraction | exposed |",
      "| ---: | ---: | ---: | ---: | ---: | ---: |",
      (
        f"| {args.compute_ms:.6f} ms | {args.comm_ms:.6f} ms | "
        f"{args.serial_ms:.6f} ms | {args.candidate_ms:.6f} ms | "
        f"{hidden_fraction:.2%} | {exposed_ms:.6f} ms |"
      ),
      "",
      "## 决策",
      "",
      f"`{decision}`",
      "",
      reason,
    ]
    args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
