"""Human-readable result aggregation and review-queue generation."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path

from .metrics import bootstrap_ci, icc_consistency
from .results import ScoreRecord

METRICS: tuple[str, ...] = (
    "trait_fidelity",
    "behavior_validity",
    "truthfulness",
    "safety",
    "sycophancy",
)
OPTIONAL_METRICS: tuple[str, ...] = ("capability_retention",)
REPORT_METRICS: tuple[str, ...] = (*METRICS, *OPTIONAL_METRICS)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(records: list[ScoreRecord]) -> dict[str, dict[str, float]]:
    """Aggregate scores by method without hiding missing groups."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for metric in METRICS:
            buckets[record.prediction.method][metric].append(float(getattr(record, metric)))
        if record.capability_retention is not None:
            buckets[record.prediction.method]["capability_retention"].append(
                float(record.capability_retention)
            )
    return {
        method: {metric: round(_average(values), 6) for metric, values in metrics.items()}
        for method, metrics in buckets.items()
    }


def aggregate_by_family(records: list[ScoreRecord]) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate scores by method and scenario family for failure localization."""
    grouped: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in records:
        grouped[f"{record.prediction.method}/{record.prediction.family}"].append(record)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for group, rows in grouped.items():
        result[group] = aggregate(rows)
    return result


def _table(summary: dict[str, dict[str, float]]) -> str:
    header = (
        "| method | "
        + " | ".join(REPORT_METRICS)
        + " |\n|---"
        + "|---:" * len(REPORT_METRICS)
        + "|\n"
    )
    rows = [header]
    for method in sorted(summary):
        values = " | ".join(
            f"{summary[method][metric]:.3f}" if metric in summary[method] else "NA"
            for metric in REPORT_METRICS
        )
        rows.append(f"| {method} | {values} |\n")
    return "".join(rows)


def _invariance_summary(records: list[ScoreRecord]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for record in records:
        key = (
            record.prediction.method,
            record.prediction.scenario_id,
            record.prediction.sample_index,
        )
        grouped[key][record.prediction.prompt_variant] = record.behavior_validity
    result: dict[str, dict[str, float | int | None]] = {}
    for method in sorted({record.prediction.method for record in records}):
        method_groups = [
            variants
            for (group_method, _scenario_id, _sample_index), variants in grouped.items()
            if group_method == method
        ]
        common_variants = (
            set.intersection(*(set(variants) for variants in method_groups))
            if method_groups
            else set()
        )
        variant_names = tuple(sorted(common_variants))
        matrices = [
            [variants[name] for name in variant_names]
            for variants in method_groups
            if len(variant_names) >= 2
        ]
        result[method] = {
            "icc": round(icc_consistency(matrices), 6) if len(matrices) >= 2 else None,
            "n_groups": len(matrices),
            "n_variants": len(variant_names),
        }
    return result


def _confidence_intervals(
    records: list[ScoreRecord], rounds: int = 1000
) -> dict[str, dict[str, dict[str, float | int]]]:
    clustered: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        for metric in REPORT_METRICS:
            value = getattr(record, metric)
            if value is not None:
                clustered[(record.prediction.method, record.prediction.scenario_id)][metric].append(
                    float(value)
                )
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (method, _scenario_id), metrics in clustered.items():
        for metric, values in metrics.items():
            buckets[method][metric].append(_average(values))
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for method, metrics in buckets.items():
        output[method] = {}
        for metric, values in metrics.items():
            low, high = bootstrap_ci(values, rounds=rounds)
            output[method][metric] = {
                "mean": round(_average(values), 6),
                "low": round(low, 6),
                "high": round(high, 6),
                "n": len(values),
            }
    return output


def _counterfactual_summary(records: list[ScoreRecord]) -> dict[str, dict[str, float | int]]:
    target_scores: dict[tuple[str, str, tuple[tuple[str, int], ...]], list[float]] = defaultdict(
        list
    )
    for record in records:
        group = record.prediction.counterfactual_group
        if not group:
            continue
        target = tuple(sorted(record.prediction.target.items()))
        target_scores[(record.prediction.method, group, target)].append(record.behavior_validity)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (method, group, _target), values in target_scores.items():
        grouped[(method, group)].append(_average(values))
    output: dict[str, dict[str, float | int]] = {}
    methods = {record.prediction.method for record in records}
    for method in sorted(methods):
        gaps = [
            max(values) - min(values)
            for (name, _group), values in grouped.items()
            if name == method and len(values) >= 2
        ]
        output[method] = {
            "mean_abs_behavior_gap": round(_average(gaps), 6) if gaps else 0.0,
            "n_groups": len(gaps),
        }
    return output


def _selectivity_summary(records: list[ScoreRecord]) -> dict[str, dict[str, float | int]]:
    """Measure target-trait change relative to non-target leakage in paired cases."""
    grouped: dict[tuple[str, str], list[ScoreRecord]] = defaultdict(list)
    for record in records:
        group = record.prediction.counterfactual_group
        if group:
            grouped[(record.prediction.method, group)].append(record)
    output: dict[str, dict[str, float | int]] = {}
    for method in sorted({record.prediction.method for record in records}):
        ratios: list[float] = []
        for (name, _group), rows in grouped.items():
            if name != method or len(rows) < 2:
                continue
            first, second = rows[0], rows[1]
            target_name = next(
                (
                    trait
                    for trait in first.prediction.target
                    if first.prediction.target[trait] != second.prediction.target.get(trait)
                ),
                None,
            )
            if target_name is None:
                continue
            if target_name not in first.trait_scores or target_name not in second.trait_scores:
                continue
            target_delta = abs(first.trait_scores[target_name] - second.trait_scores[target_name])
            leakage = sum(
                abs(first.trait_scores.get(trait, 0.0) - second.trait_scores.get(trait, 0.0))
                for trait in first.prediction.target
                if trait != target_name
            )
            ratios.append(target_delta / (1e-6 + leakage))
        output[method] = {
            "mean_target_to_leakage": round(_average(ratios), 6) if ratios else 0.0,
            "n_pairs": len(ratios),
        }
    return output


def render_markdown(records: list[ScoreRecord]) -> str:
    """Render an auditable Markdown report with summary and flagged cases."""
    summary = aggregate(records)
    flagged = [record for record in records if record.rule_flags or record.judge_confidence < 0.8]
    family_rows = []
    for group, values in sorted(aggregate_by_family(records).items()):
        method_values = next(iter(values.values()))
        family_rows.append(
            f"| {group} | {method_values.get('behavior_validity', 0.0):.3f} | "
            f"{method_values.get('truthfulness', 0.0):.3f} | "
            f"{method_values.get('safety', 0.0):.3f} |"
        )
    lines = (
        [
            "# Persona-RL Experiment Report",
            "",
            f"Total scored predictions: **{len(records)}**",
            "",
            "## Summary",
            "",
            _table(summary),
            "",
            "## Failure localization",
            "",
            "| method/family | behavior_validity | truthfulness | safety |",
            "|---|---:|---:|---:|",
        ]
        + family_rows
        + [
            "",
            "## Method provenance",
            "",
            "Every row retains run_id, model_id, prompt variant, temperature, and judge "
            "metadata and rubric version. See `predictions.jsonl` and `scores.jsonl`.",
            "",
            f"## Review queue ({len(flagged)})",
            "",
            "| id | method | family | variant | sample | flags | confidence | response preview |",
            "|---|---|---|---|---:|---|---:|---|",
        ]
    )
    for record in flagged[:200]:
        preview = " ".join(record.prediction.response.split())[:120].replace("|", "\\|")
        lines.append(
            f"| {record.prediction.scenario_id} | {record.prediction.method} | "
            f"{record.prediction.family} | {record.prediction.prompt_variant} | "
            f"{record.prediction.sample_index} | "
            f"{','.join(record.rule_flags) or 'low_confidence'} | "
            f"{record.judge_confidence:.2f} | {preview} |"
        )
    invariance = _invariance_summary(records)
    lines.extend(
        [
            "",
            "## Prompt invariance",
            "",
            "| method | ICC(2,1) | paired groups | common variants |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, values in invariance.items():
        icc = "NA" if values["icc"] is None else f"{values['icc']:.3f}"
        lines.append(f"| {method} | {icc} | {values['n_groups']} | {values['n_variants']} |")
    lines.extend(
        [
            "",
            "## Bootstrap intervals",
            "",
            "The JSON report stores deterministic percentile 95% CIs for every available metric.",
            "",
        ]
    )
    for method, metrics in _confidence_intervals(records).items():
        rendered = ", ".join(
            f"{metric}={values['mean']:.3f} [{values['low']:.3f}, {values['high']:.3f}]"
            for metric, values in metrics.items()
        )
        lines.append(f"- {method}: {rendered}")
    lines.extend(
        [
            "",
            "## Counterfactual target sensitivity",
            "",
            "| method | mean behavior gap across target pairs | paired groups |",
            "|---|---:|---:|",
        ]
    )
    for method, values in _counterfactual_summary(records).items():
        lines.append(f"| {method} | {values['mean_abs_behavior_gap']:.3f} | {values['n_groups']} |")
    return "\n".join(lines) + "\n"


def render_html(records: list[ScoreRecord]) -> str:
    """Render a standalone HTML report suitable for opening on a server."""
    summary = aggregate(records)
    header = "".join(f"<th>{html.escape(metric)}</th>" for metric in REPORT_METRICS)
    rows = []
    for method in sorted(summary):
        cells = "".join(
            f"<td>{summary[method][metric]:.3f}</td>"
            if metric in summary[method]
            else "<td>NA</td>"
            for metric in REPORT_METRICS
        )
        rows.append(f"<tr><th>{html.escape(method)}</th>{cells}</tr>")
    flagged = [record for record in records if record.rule_flags or record.judge_confidence < 0.8][
        :200
    ]
    review_rows = []
    for record in flagged:
        review_rows.append(
            "<tr>"
            f"<td>{html.escape(record.prediction.scenario_id)}</td>"
            f"<td>{html.escape(record.prediction.method)}</td>"
            f"<td>{html.escape(record.prediction.family)}</td>"
            f"<td>{html.escape(record.prediction.prompt_variant)}</td>"
            f"<td>{record.prediction.sample_index}</td>"
            f"<td>{html.escape(record.prediction.response[:240])}</td>"
            f"<td>{html.escape(','.join(record.rule_flags) or 'low_confidence')}</td>"
            "</tr>"
        )
    invariance_rows = []
    for method, values in _invariance_summary(records).items():
        icc = "NA" if values["icc"] is None else f"{float(values['icc']):.3f}"
        invariance_rows.append(
            f"<tr><th>{html.escape(method)}</th><td>{icc}</td>"
            f"<td>{values['n_groups']}</td><td>{values['n_variants']}</td></tr>"
        )
    style = (
        "<style>body{font:15px system-ui;max-width:1200px;margin:2rem auto}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #ddd;padding:.45rem;text-align:left}"
        "th{background:#f3f5f7}code{white-space:pre-wrap}</style>"
    )
    summary_table = f"<table><tr><th>method</th>{header}</tr>{''.join(rows)}</table>"
    invariance_table = (
        "<table><tr><th>method</th><th>ICC(2,1)</th><th>paired groups</th>"
        f"<th>common variants</th></tr>{''.join(invariance_rows)}</table>"
    )
    review_table = (
        "<table><tr><th>id</th><th>method</th><th>family</th><th>variant</th>"
        f"<th>sample</th><th>response</th><th>flags</th></tr>{''.join(review_rows)}</table>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Persona-RL report</title>"
        f"{style}</head><body><h1>Persona-RL Experiment Report</h1>"
        f"<p>Scored predictions: {len(records)}</p><h2>Summary</h2>{summary_table}"
        f"<h2>Prompt invariance</h2>{invariance_table}"
        f"<h2>Review queue</h2>{review_table}</body></html>"
    )


def write_report(input_path: str, output_dir: str) -> None:
    """Write Markdown, HTML, CSV, and review JSONL artifacts."""
    from .results import read_models

    records = read_models(input_path, ScoreRecord)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "report-v1",
                "input_path": str(input_path),
                "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                "n_records": len(records),
                "rubric_versions": sorted({record.rubric_version for record in records}),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (destination / "report.md").write_text(render_markdown(records), encoding="utf-8")
    (destination / "report.html").write_text(render_html(records), encoding="utf-8")
    (destination / "summary.json").write_text(
        json.dumps(
            {
                "n": len(records),
                "by_method": aggregate(records),
                "by_family": aggregate_by_family(records),
                "prompt_invariance": _invariance_summary(records),
                "counterfactual_sensitivity": _counterfactual_summary(records),
                "selectivity": _selectivity_summary(records),
                "confidence_intervals": _confidence_intervals(records),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with (destination / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *REPORT_METRICS])
        for method, values in sorted(aggregate(records).items()):
            writer.writerow([method, *(values.get(metric, "NA") for metric in REPORT_METRICS)])
    flagged = [record for record in records if record.rule_flags or record.judge_confidence < 0.8]
    with (destination / "review_queue.jsonl").open("w", encoding="utf-8") as handle:
        for record in flagged:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
