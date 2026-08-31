# 正式实验运行手册

本手册规定一轮可进入论文表格的实验必须执行的步骤。任何跳过 provenance、双 judge 或发布前校验的运行都只能视为 pilot。

## 1. 固定实验条件

在服务器上记录：GPU 型号与数量、驱动、CUDA、Python、`uv.lock` hash、基础模型 revision、两个 judge revision、生成模型 revision和代码 commit。每个训练 seed 使用不同目录，禁止覆盖 checkpoint。

主实验至少使用两个不同模型家族，每个方法三个 seed。Base、SFT、Direct-DPO、CB-DPO 使用相同 tokenizer、最大长度、LoRA rank、训练 token 预算、解码温度和测试 prompt。测试集不得参与 reward threshold、prompt 或 checkpoint 选择。

## 2. 数据构造

```bash
cd persona_rl
export PYTHONPATH="$PWD/src"
uv run scripts/generate_scenarios.py --output data/raw/scenarios.jsonl \
  --count 6000 --capability-count 600 --seed 7 --split-strategy family_holdout
uv run scripts/audit_cli.py data/raw/scenarios.jsonl \
  --output-path data/processed/scenario_audit.jsonl --limit 120
```

每个 counterfactual group 必须只改变一个 trait；其他 trait 为 0。测试按 family 留出，而不是随机句子划分。正式数据使用独立 LLM 扩写候选，`build_preferences.py` 的占位 pair 禁止用于论文训练。

另外下载外部 frozen evaluation（不参与训练）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv run scripts/fetch_external_datasets.py persona_chat --output-dir data/raw/external
uv run scripts/fetch_external_datasets.py empathetic_dialogues --output-dir data/raw/external
uv run scripts/fetch_external_datasets.py prosocial_dialog --output-dir data/raw/external
uv run scripts/fetch_external_datasets.py truthful_qa --output-dir data/raw/external
```

每个外部源必须保留 `manifest.json`、许可证和下载 revision。PersonaChat 的 GitHub/ParlAI 版本需用 `gh repo clone` 并记录 commit。外部数据先通过 dataset-specific adapter 转成评测 schema；没有人格标签的数据只能报告为行为、同理心、安全或真实性外部效度，不能当作人格训练金标准。

## 3. 双 Judge 构造偏好

```bash
uv run scripts/expand_with_llm.py data/raw/scenarios.jsonl \
  --output-path data/processed/candidate_pairs.jsonl --model <generator> \
  --candidates-per-scenario 4 --split train --resume
uv run scripts/judge_pairs.py data/processed/candidate_pairs.jsonl \
  --output-path data/processed/judged_pairs.jsonl \
  --model <judge-a> --second-model <judge-b>
```

报告候选总数、保留率、tie 率、judge disagreement、各阈值拒绝率和最终 trait/family 分布。两个 judge 均不得与训练策略使用相同 checkpoint。

## 4. 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 scripts/train_dpo.py \
  data/processed/judged_pairs.jsonl --model <base-model> --model-revision <revision> \
  --method direct-dpo --output artifacts/checkpoints/direct_seed7 --seed 7

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 scripts/train_dpo.py \
  data/processed/judged_pairs.jsonl --model <base-model> --model-revision <revision> \
  --method pc-dpo --output artifacts/checkpoints/cb_dpo_seed7 --seed 7
```

CB-DPO 在数据阶段执行硬约束过滤，在 batch 阶段联合优化 DPO、criterion、invariance、truth、safety、uncertainty 和 reference KL。训练完成后检查 `training_config.json`，并保存 checkpoint hash。

## 5. 预测与正式评分

```bash
uv run scripts/run_experiment.py data/raw/scenarios.jsonl --backend hf --split test \
  --methods base,direct_dpo,pc_dpo \
  --model-map base=<base>,direct_dpo=<direct>,pc_dpo=<cb-dpo> \
  --samples 3 --temperature 0.7 --scorer none

uv run scripts/llm_score_outputs.py artifacts/experiment/<run>/predictions.jsonl \
  data/raw/scenarios.jsonl --output artifacts/experiment/<run>/scores.jsonl \
  --model <judge-c> --second-model <judge-d>
```

## 5a. Trait × Style 正交实验

在正式预测前固定一个 2×6 因子矩阵：每个隐藏行为情境复制到两个 trait 档位（高/低）和六种 style（`neutral`、`warm/polite`、`blunt/direct`、`formal`、`terse`、`conversational`）。必须包含高宜人性×`blunt/direct`、低宜人性×`warm/polite`等冲突单元。生成器可见控制字段，但输出不得泄露字段名；judge 只接收情境、回答和不含 style 名称的 rubric。

所有方法共享相同 cell、解码预算和随机种子。按 `scenario_family × trait_level × style` 聚合，拟合预注册模型 `Y ~ trait * style + (1|family) + (1|scenario)`，分别对行为效度、trait 分数、style 分数和安全/真实性评分建模。输出每个 cell 的均值/95% CI、trait/style/交互系数、跨 style ICC、行为 style-slope 和冲突单元反转率。

必须对同一预测文件执行 `style_normalize.py` 和第二轮 judge，比较归一化前后的行为效度与交互项。发布门槛是：CB-DPO 的跨 style 最小 trait effect 为正且 CI 不跨 0，行为和安全结果的 style-slope 在预注册等效性界内，且冲突反转率低于 Direct-DPO。这样才能把“学到行为层人格”与“学到固定话术”区分开。

训练 judge 与测试 judge 分离。保存逐样本评分、原始 response、rubric version、置信度、结构化 option、gold match 和 probability。关键词 scorer 只允许做 CI/smoke。

## 6. 表面风格中介分析

```bash
uv run scripts/style_normalize.py artifacts/experiment/<run>/predictions.jsonl \
  --output artifacts/experiment/<run>/normalized_predictions.jsonl --model <rewriter>
uv run scripts/llm_score_outputs.py artifacts/experiment/<run>/normalized_predictions.jsonl \
  data/raw/scenarios.jsonl --output artifacts/experiment/<run>/normalized_scores.jsonl \
  --model <judge-c> --second-model <judge-d>
```

分别报告原始和归一化输出上的行为效度。只有当 style-normalized 后 Direct-DPO 的收益下降而 CB-DPO 的结构化行为收益保留时，才支持“表面实现中介”的解释。

## 7. 盲法人审

```bash
PERSONA_RL_REVIEWER=<id> uv run scripts/human_score.py \
  artifacts/experiment/<run>/predictions.jsonl --scenarios data/raw/scenarios.jsonl \
  --output artifacts/experiment/<run>/report/human_scores.<id>.jsonl --limit 120
```

至少两位评分者使用独立文件。样本按 method/family 分层，评分者看不到 method。报告评分者间一致性、人工与 judge 相关性、争议率及分层失败案例。

## 8. 统计与发布门槛

以 scenario 为 cluster，使用同一 scenario/variant/sample 的配对比较。报告 bootstrap 95% CI、效应量、ICC、AUROC、Brier、ECE、安全/能力等效性检验。不能把同一模型的多次 sampling 当作独立模型 seed。

发布前必须通过：

```bash
uv run scripts/compare_methods.py artifacts/experiment/<run>/scores.jsonl \
  --output artifacts/experiment/<run>/report/pairwise.csv
uv run scripts/validate_artifacts.py artifacts/experiment/<run> \
  --expected-methods base,direct_dpo,pc_dpo \
  --expected-variants canonical,paraphrase,minimal,formal,terse,conversational
```

主方法成功标准：相对 Direct-DPO 的隐藏行为效度配对 CI 不跨 0；跨风格 ICC 和 trait selectivity 提升；第二 judge 与盲法人审方向一致；truth/safety 不退化；通用能力下降不超过预注册阈值。
