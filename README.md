# CB-DPO: 行为有效、风格不变的人格化行为控制

本仓库实现一套可复现的 LLM 人格化行为研究管线。研究对象不是模型是否拥有人的内在人格，而是：给定人格目标向量后，模型是否在未见情境中表现出可验证、可迁移、跨风格稳定且不牺牲真实性与安全性的行为倾向。

## 核心方法

主方法是 **offline constraint-weighted PC-DPO / CB-DPO**。每条样本包含情境、单 trait 反事实目标、结构化行为选项和 gold option。独立 judge 为候选回答产生 trait、hidden criterion、style invariance、truthfulness、safety、uncertainty 多头分数。训练目标为：

\[
\ell = w(r)\ell_{DPO}+0.25(1-r_c)+0.25(1-r_i)+0.25u+0.05KL(\pi_\theta\|\pi_{ref})
\]

`pc_dpo_trainer.py` 将这些项接入 TRL `compute_loss` 并反向传播；`judge_pairs.py` 负责双 judge 过滤与 reward 缓存。当前实现是离线 DPO，不是在线 PPO。

## 证据链

论文主张只有在以下证据同时成立时才成立：

1. Direct-DPO 的人格 rubric/问卷分数上升，但隐藏行为效度不稳定；
2. CB-DPO 在单 trait 反事实、未见情境和跨风格测试中提高行为效度与 target-to-leakage；
3. style normalization 后 CB-DPO 优势保留，而表面风格收益显著下降；
4. 第二个 judge 和盲法人工审计不推翻结论；
5. truthfulness、safety 和通用能力不超过预注册退化阈值。

## 快速开始

```bash
cd persona_rl
uv sync
export PYTHONPATH="$PWD/src"
uv run scripts/generate_scenarios.py --output data/raw/scenarios.jsonl --count 1800 --capability-count 300 --seed 7
uv run scripts/expand_with_llm.py data/raw/scenarios.jsonl --output-path data/processed/candidate_pairs.jsonl --split train --resume
uv run scripts/judge_pairs.py data/processed/candidate_pairs.jsonl --output-path data/processed/judged_pairs.jsonl --model <judge> --second-model <second-judge>
uv run scripts/train_dpo.py data/processed/judged_pairs.jsonl --method pc-dpo --model <base-model> --output artifacts/checkpoints/pc_dpo_seed7
```

训练依赖：`uv sync --extra train`。多 GPU 使用 `accelerate launch`，每个 seed 使用独立输出目录。正式运行必须固定模型 revision、数据 hash、judge revision、CUDA/驱动和命令日志。

## 评测

禁止用关键词匹配作为论文指标。正式评测使用独立 LLM judge：

```bash
uv run scripts/run_experiment.py data/raw/scenarios.jsonl --backend hf \
  --model-map base=<base>,direct_dpo=<direct>,pc_dpo=<pc> \
  --methods base,direct_dpo,pc_dpo --scorer none --split test
uv run scripts/llm_score_outputs.py artifacts/experiment/<run>/predictions.jsonl \
  data/raw/scenarios.jsonl --output artifacts/experiment/<run>/scores.jsonl \
  --model <judge> --second-model <second-judge>
uv run scripts/render_report.py artifacts/experiment/<run>/scores.jsonl --output-dir artifacts/experiment/<run>/report
```

主指标是 hidden behavior validity、AUROC/Brier/calibration、prompt/style ICC、单 trait effect、非目标 leakage、truthfulness、safety、sycophancy 和 capability retention。`style_normalize.py` 生成中性专业改写后，应重新执行 LLM judge 比较归一化前后效度。

## 盲法人审

人工评测只做小规模外部效度审计，不参与训练或调参。程序显示情境、rubric 和结构化选项，随机打乱回答并隐藏 method：

```bash
PERSONA_RL_REVIEWER=alice uv run scripts/human_score.py \
  artifacts/experiment/<run>/predictions.jsonl --scenarios data/raw/scenarios.jsonl \
  --output artifacts/experiment/<run>/report/human_scores.jsonl --limit 120
```

每个样本评分 trait fidelity、behavior validity、truthfulness、safety 和 confidence，建议按 method/family 分层抽取 100--200 条。

## 目录

- `persona_rl/src/persona_rl/`：typed schema、推理、指标、约束 reward 和自定义 trainer。
- `persona_rl/scripts/`：数据生成、候选扩写、双 judge、训练、评测、报告、style normalization、人审和 baseline wrapper。
- `persona_rl/configs/`：实验矩阵与固定外部 baseline provenance。
- `persona_rl/REFERENCES.md`：论文、源码、数据集和 BibTeX。
- `CB_DPO_方法与评测实现规范.md`：方法与评测契约。
- `docs/archive/`：早期研究草稿，仅供历史参考。

外部算法源码通过 GitHub CLI 获取并保留原始仓库；它们只在明确可复现时进入比较，PersonaForge 是 related work，不是 baseline。发布前运行：

```bash
uv run scripts/validate_artifacts.py artifacts/experiment/<run> \
  --expected-methods base,direct_dpo,pc_dpo \
  --expected-variants canonical,paraphrase,minimal,formal,terse,conversational
```

完整实验设计、统计假设、资源预算和限制见 [方法与实验规格](方法与实验规格_Psychometrics_Constrained_DPO.md)。
