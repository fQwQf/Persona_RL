# 1.5B server pilot: results and interpretation

This report records the first end-to-end run on `WHU140piROOT`. It is an
engineering and feasibility pilot, not a paper result. All automatic scores
below come from the same structured Llama-3.1-8B judge; no keyword scorer was
used for the reported numbers.

## Run manifest

The remote host was `WHU140piROOT`, using
`Qwen/Qwen2.5-1.5B-Instruct` from ModelScope. The policy runs used LoRA
(rank 16, alpha 32), one seed (`7`), three epochs, batch size 1,
gradient accumulation 1, learning rate `5e-6`, maximum length 512, and no
QLoRA. Direct-DPO and PC-DPO used exactly the same accepted preference file
and optimization settings.

The scenario file contained 240 bilingual rows:

| split | rows | language | use |
|---|---:|---:|---|
| train | 168 | 84 zh / 84 en | candidate generation and preference mining |
| validation | 36 | 18 zh / 18 en | reserved |
| test | 36 | 18 zh / 18 en | held-out pilot evaluation |

Candidates were generated for all 168 training rows. A strict Llama-8B
primary judge accepted 12 pairs after requiring trait, truth, and safety
thresholds of 0.7. The other 156 rows were rejected or uncertain. A dual
judge run using Qwen-1.5B as the second judge accepted zero rows because the
small judge was systematically uncertain/disagreeing; it was not used to
train the reported checkpoints.

Remote artifact paths:

```text
/tmp/persona_pilot_scenarios.jsonl
/tmp/persona_pilot_candidates.jsonl
/tmp/persona_pilot_judged_primary.jsonl
/tmp/pilot_direct_strict
/tmp/pilot_pc_strict2
/tmp/pilot_eval_test_all.jsonl
/tmp/pilot_eval_test_scores.jsonl
/tmp/pilot_eval_test_pairwise.csv
/tmp/pilot_eval_test_pairwise.md
```

The held-out evaluation used 36 test scenarios, two prompt variants
(`canonical`, `paraphrase`), and one deterministic sample per cell. Thus the
judge scored 216 predictions: 72 for each of Base, Direct-DPO, and PC-DPO.

## Training sanity check

Both trainers completed and wrote valid LoRA adapters.

| method | train rows | steps | runtime | final train loss | final reward accuracy |
|---|---:|---:|---:|---:|---:|
| Direct-DPO | 12 | 36 | 85.6 s | 0.6656 | 1.00 |
| PC-DPO | 12 | 36 | 78.4 s | 0.6654 | 1.00 |

The reward accuracy is an in-training DPO diagnostic, not an evaluation
metric. With only 12 pairs, memorization is a serious possibility.

## Held-out judge scores

The primary aggregate is the unweighted mean of trait fidelity, behavior
validity, truthfulness, and safety. Higher is better for every metric except
sycophancy, where lower is better.

| method | trait fidelity | behavior validity | truthfulness | safety | sycophancy | aggregate |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.515 | 0.603 | 0.618 | 0.846 | 0.121 | 0.645 |
| Direct-DPO | 0.600 | 0.575 | 0.633 | 0.875 | 0.075 | 0.671 |
| PC-DPO | 0.592 | 0.650 | 0.738 | 0.900 | 0.067 | 0.720 |

Paired deltas over the same scenario and prompt variant were:

| comparison | behavior validity | safety | sycophancy |
|---|---:|---:|---:|
| Direct-DPO - Base | -0.028 [-0.069, 0.025] | +0.029 [-0.017, 0.090] | -0.046 [-0.103, -0.003] |
| PC-DPO - Base | +0.047 [-0.019, 0.112] | +0.054 [+0.008, 0.112] | -0.054 [-0.111, -0.010] |
| PC-DPO - Direct-DPO | +0.075 [+0.024, +0.129] | +0.025 [+0.006, +0.046] | -0.008 [-0.014, -0.003] |

The bracketed intervals are the existing dependency-free bootstrap intervals
over paired scenario means. For the four-metric aggregate, the paired mean
delta was `+0.074` for PC-DPO over Base (bootstrap 95% interval
`[+0.022, +0.133]`) and `+0.049` over Direct-DPO (`[+0.016, +0.087]`).

## Invariance diagnostic

Aggregate score by prompt variant:

| method | canonical | paraphrase | paraphrase - canonical |
|---|---:|---:|---:|
| Base | 0.769 | 0.522 | -0.247 |
| Direct-DPO | 0.729 | 0.613 | -0.117 |
| PC-DPO | 0.721 | 0.719 | -0.002 |

This is directionally consistent with the method hypothesis: PC-DPO gives up
some canonical score while maintaining the target behavior under a prompt
paraphrase. It is not yet a Trait x Style result; all pilot scenarios have
`style_family=neutral`, so the full six-style orthogonal matrix still needs to
be run.

## What this pilot establishes

It establishes that the full data -> strict judge -> Direct-DPO/PC-DPO ->
held-out generation -> structured judge pipeline runs on a single 1.5B policy
and produces a measurable signal. The signal is in the predicted direction:
PC-DPO improves the constraint-heavy metrics and paraphrase stability more
than Direct-DPO.

It does **not** establish that the policy has learned a human-like
personality, or that PC-DPO is superior at paper quality. The main reasons are
the 12-pair training set, one seed, three held-out scenario families, reuse of
the Llama-8B judge for mining and evaluation, and the absence of the required
multi-style conflict cells. The evaluation prompts also expose the structured
target code to every method, so this pilot is not a clean test of hidden
persona inference.

## Decision for the next run

The method passes the engineering go/no-go gate and is worth scaling. The
next run should not change the objective based on this pilot. It should first
create the train/validation/test split before candidate generation, generate
at least 1k--5k accepted pairs with a stronger independent judge, add the full
Trait x Style conflict matrix, and run three seeds. The final evaluation must
use a judge not involved in preference mining plus a small blinded human audit.

For a faster formal run, use vLLM for batched policy inference. The serial HF
backend used here took roughly 8--11 minutes per 72-output method; that is
appropriate for correctness checks but not for the full experiment.

