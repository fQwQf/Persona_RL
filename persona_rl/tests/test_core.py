from persona_rl.inference import InferenceConfig, InferenceEngine, build_prompt
from persona_rl.metrics import agreement, bootstrap_ci, icc_consistency, pearson, variance
from persona_rl.constraints import ConstraintReward, pc_dpo_scalar_loss
from persona_rl.reporting import write_report
from persona_rl.results import PredictionRecord, ScoreRecord, parse_method, write_models
from persona_rl.schema import HiddenTask, Scenario, TargetTraits, parse_jsonl


def test_metrics_on_paired_values() -> None:
    assert agreement([1, 0], [1, 1]) == 0.5
    assert pearson([1.0, 2.0], [2.0, 4.0]) == 1.0
    assert variance([1.0, 3.0]) == 1.0
    low, high = bootstrap_ci([0.0, 1.0], rounds=100)
    assert 0 <= low <= high <= 1
    assert icc_consistency([[1.0, 1.0], [0.0, 0.0]]) == 1.0


def test_generated_jsonl_is_typed(tmp_path) -> None:
    path = tmp_path / "scenario.jsonl"
    path.write_text(
        '{"id":"x","family":"f","split":"test","target_z":{"conscientiousness":1,"agreeableness":0,"honesty_humility":1},"situation":"s","hidden_task":{"type":"binary_decision","gold_behavior":1},"counterfactual_group":"c"}\n',
        encoding="utf-8",
    )
    records = parse_jsonl(str(path))
    assert records[0].target_z.conscientiousness == 1


def test_method_parser_rejects_unpinned_method() -> None:
    assert parse_method("pc_dpo") == "pc_dpo"
    try:
        parse_method("unknown")
    except ValueError:
        return
    raise AssertionError("unknown method was accepted")


def test_report_writes_machine_and_human_outputs(tmp_path) -> None:
    prediction = PredictionRecord(
        run_id="r",
        method="pc_dpo",
        model_id="m",
        scenario_id="s",
        family="f",
        target={"conscientiousness": 1, "agreeableness": 0, "honesty_humility": 1},
        temperature=0.7,
        sample_index=0,
        prompt="p",
        response="r",
        latency_ms=1,
    )
    score = ScoreRecord(
        prediction=prediction,
        trait_fidelity=1,
        behavior_validity=1,
        truthfulness=1,
        safety=1,
        sycophancy=0,
        judge_confidence=1,
    )
    scores = tmp_path / "scores.jsonl"
    write_models(str(scores), [score])
    output = tmp_path / "report"
    write_report(str(scores), str(output))
    assert (output / "report.md").exists()
    assert (output / "report.html").exists()
    assert (output / "summary.json").exists()


def test_prompt_variants_are_distinct_and_recorded() -> None:
    scenario = Scenario(
        id="variant",
        family="f",
        split="test",
        target_z=TargetTraits(conscientiousness=1, agreeableness=0, honesty_humility=1),
        situation="请给出一个可执行的建议。",
        behavior_rubric=("建议",),
        hidden_task=HiddenTask(type="binary_decision", gold_behavior=1),
        counterfactual_group="cf",
    )
    prompts = {
        build_prompt(scenario, "base", "system", variant)
        for variant in ("canonical", "paraphrase", "minimal")
    }
    assert len(prompts) == 3
    engine = InferenceEngine(InferenceConfig("base", "m", "dry_run", "", "", 0.7, 32, "system"))
    prediction = engine.generate(scenario, 0, "paraphrase")
    assert prediction.prompt_variant == "paraphrase"


def test_counterfactual_changes_one_trait_and_has_machine_task_options(tmp_path) -> None:
    """Generated counterfactual pairs must isolate a single causal trait."""
    import subprocess
    from pathlib import Path

    path = tmp_path / "scenarios.jsonl"
    project = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["uv", "run", "scripts/generate_scenarios.py", "--output", str(path), "--count", "20"],
        check=True,
        cwd=project,
    )
    rows = parse_jsonl(str(path))
    pair = [row for row in rows if row.counterfactual_group == "cf_test_000000"]
    assert len(pair) == 2
    differences = [
        trait for trait in pair[0].target_z.model_dump()
        if getattr(pair[0].target_z, trait) != getattr(pair[1].target_z, trait)
    ]
    assert len(differences) == 1
    assert pair[0].hidden_task.target_trait == differences[0]
    assert pair[0].hidden_task.options
    assert pair[0].hidden_task.gold_option != pair[1].hidden_task.gold_option


def test_pc_dpo_penalizes_uncertain_reward() -> None:
    """Uncertainty and weak constraints must increase the optimization loss."""
    strong = ConstraintReward(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    weak = ConstraintReward(1.0, 0.2, 0.2, 1.0, 1.0, 0.8)
    assert pc_dpo_scalar_loss(1.0, 0.0, weak) > pc_dpo_scalar_loss(1.0, 0.0, strong)
