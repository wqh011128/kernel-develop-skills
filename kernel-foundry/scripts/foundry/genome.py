from __future__ import annotations

from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .common import dump_json, load_json, now_utc, require, require_fields, stable_id


def _matches_constraint(genome: dict[str, Any], constraint: dict[str, Any]) -> bool:
    require_fields(constraint, ("if", "then"), "genome constraint")
    antecedent = constraint["if"]
    consequent = constraint["then"]
    applies = all(genome.get(key) == value for key, value in antecedent.items())
    return not applies or all(genome.get(key) == value for key, value in consequent.items())


def _matches_gene_rules(genome: dict[str, Any], gene_rules: dict[str, Any]) -> bool:
    for gene, rules in gene_rules.items():
        if gene not in genome:
            continue
        value = genome[gene]
        require(isinstance(rules, dict), f"gene_rules.{gene} must be an object")
        if "min" in rules and value < rules["min"]:
            return False
        if "max" in rules and value > rules["max"]:
            return False
        if "multiple_of" in rules:
            divisor = rules["multiple_of"]
            require(divisor > 0, f"gene_rules.{gene}.multiple_of must be positive")
            if value % divisor != 0:
                return False
    return True


def propose_mutations(spec_path: Path, output_path: Path, limit: int) -> dict[str, Any]:
    require(limit > 0, "limit must be positive")
    spec = load_json(spec_path)
    require(isinstance(spec, dict), "genome spec must be a JSON object")
    require_fields(spec, ("base", "search_space"), "genome spec")
    base = spec["base"]
    search_space = spec["search_space"]
    require(isinstance(base, dict) and isinstance(search_space, dict), "base and search_space must be objects")
    seen = set(spec.get("seen", []))
    constraints = spec.get("constraints", [])
    gene_rules = spec.get("gene_rules", {})
    parent_id = spec.get("parent_id", stable_id(base, "genome-"))
    proposals = []
    rejected_by_constraints = 0
    for gene in sorted(search_space):
        require(isinstance(search_space[gene], list), f"search_space.{gene} must be a list")
        for value in search_space[gene]:
            if base.get(gene) == value:
                continue
            genome = {**base, gene: value}
            fingerprint = stable_id(genome, "genome-")
            if fingerprint in seen:
                continue
            if not _matches_gene_rules(genome, gene_rules) or not all(
                _matches_constraint(genome, rule) for rule in constraints
            ):
                rejected_by_constraints += 1
                continue
            proposals.append(
                {
                    "id": fingerprint,
                    "parent_id": parent_id,
                    "mutation": {"gene": gene, "before": base.get(gene), "after": value},
                    "genome": genome,
                    "status": "proposed_unverified",
                }
            )
            if len(proposals) >= limit:
                break
        if len(proposals) >= limit:
            break
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_utc(),
        "parent_id": parent_id,
        "base": base,
        "mutation_policy": "one_gene_at_a_time",
        "proposals": proposals,
        "rejected_by_constraints": rejected_by_constraints,
        "acceptance_required": ["independent_correctness", "objective_metrics", "experiment_record"],
    }
    dump_json(output_path, report)
    return report
