# Persona-RL 实验工程

## 目录

- `scripts/generate_scenarios.py`：程序化生成、按 split 写入场景 JSONL。
- `scripts/build_preferences.py`：把场景转成 TRL DPO 所需的 `prompt/chosen/rejected`。
- `scripts/expand_with_llm.py`：调用 OpenAI-compatible endpoint 生成候选回答并记录原始输出。
- `scripts/judge_pairs.py`：用独立 JSON judge 过滤候选 pair，拒绝 tie 和硬约束失败样本。
- `scripts/run_inference.py`：对任意方法生成统一 `PredictionRecord` JSONL。
- `scripts/merge_predictions.py`：合并多 GPU rank shard，并拒绝重复条件。
- `scripts/score_outputs.py`：用透明规则给每个输出评分。
- `scripts/llm_score_outputs.py`：用独立结构化 judge 评分，支持断点续跑与输入哈希。
- `scripts/render_report.py`：生成 Markdown、HTML、CSV、JSON 和审计队列。
- `scripts/review_results.py`：交互式抽查低置信度/规则冲突样本。
- `scripts/human_score.py`：盲法 1–5 Likert 人工评分；不展示 method 名称，不产生训练标签。
- `scripts/style_normalize.py`：使用独立 LLM 将回答改写为中性风格，保持动作/事实不变，用于表面风格中介分析。
- `scripts/make_audit_queue.py`：按 method/family 分层抽取固定比例的人工审计样本。
- `scripts/summarize_audit.py`：汇总审计决定、reviewer 覆盖和不确定率。
- `scripts/run_experiment.py`：在同一 split 成对运行多个方法并汇总报告。
- `scripts/compare_methods.py`：输出逐情境配对 delta 和 bootstrap 区间。
- `scripts/validate_artifacts.py`：在发布前检查行对齐、重复样本、provenance 和报告完整性。
- `scripts/clone_baselines.py`：通过 GitHub CLI 克隆并校验公开 baseline commit。
- `scripts/run_official_baseline.py`：记录或执行原仓库命令，不改写原始实现。
- `scripts/normalize_external.py`：将官方 CSV/JSONL 结果转成共同预测 schema。
- `scripts/run_baseline.py`：统一记录外部方法 provenance；可选执行官方命令并直接归一化 raw 输出。
- `REFERENCES.md`：论文、源码、数据集和 BibTeX 引用清单。
- `../CB_DPO_方法与评测实现规范.md`：论文主方法、可识别性和评测实现契约。

- `scripts/audit_cli.py`：低规模交互式 rubric 审计，不产生训练标签。
- `scripts/fetch_public.py`：下载公开数据并记录 SHA-256 manifest。
- `scripts/train_dpo.py`：Direct-DPO/PC-DPO 训练入口。
- `scripts/train_sft.py`：同模型、同 token budget 的 SFT LoRA 对照。
- `scripts/evaluate.py`：聚合评测 JSONL。

已获取外部评测资源：`data/raw/PsychoBench`，固定 commit `d514fb0810f4c79571b4e13e588ffe7f5daaa24f`。它只用于 frozen evaluation，不得混入训练；其 LICENSE 和原始 README 保留在目录内。
公开 baseline 源码固定在 `external/`，完整 provenance 见 `configs/baseline_manifest.json`。
PersonaForge 仅作为最接近的 related work 和概念 reference，引用与定位见 `REFERENCES.md`；它不进入 baseline registry、统一方法运行入口或实验结果表。

## 安装

```bash
cd persona_rl
uv sync --extra train --extra judge
export PYTHONPATH="$PWD/src"
```

训练依赖（含 CUDA wheel）建议预留 8–12 GB 磁盘；7B QLoRA 还需要 CUDA、PyTorch、Transformers、TRL、PEFT、Accelerate 和 bitsandbytes。正式服务器应固定 `uv.lock`、CUDA、驱动和模型 revision。

## 数据管线

```bash
uv run scripts/generate_scenarios.py --output data/raw/scenarios.jsonl --count 1800 --capability-count 300 --seed 7 --split-strategy family_holdout
uv run scripts/audit_cli.py data/raw/scenarios.jsonl --output-path data/processed/audit.jsonl --limit 100
uv run scripts/build_preferences.py data/raw/scenarios.jsonl --output-path data/processed/preferences.jsonl
```

`build_preferences.py` 的 chosen/rejected 是 smoke-test 占位数据。正式实验必须替换为 LLM 扩写、双 judge、规则过滤后的候选对，不能把占位文本作为论文数据。

LLM 扩写（支持 OpenAI、vLLM 或其他兼容服务）：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=http://server:8000/v1/chat/completions
PYTHONPATH=src uv run scripts/expand_with_llm.py data/raw/scenarios.jsonl \
  --output-path data/processed/candidate_pairs.jsonl --model Qwen/Qwen2.5-72B-Instruct \
  --split train --resume
```

候选 pair 需要包含 `prompt、candidate_a、candidate_b、id` 字段，再运行：

```bash
PYTHONPATH=src uv run scripts/judge_pairs.py data/processed/candidate_pairs.jsonl \
  --output-path data/processed/judged_pairs.jsonl --model Qwen/Qwen2.5-72B-Instruct \
  --second-model gpt-4o-mini
```

`judge_pairs.py` 同时写出 `judged_pairs.audit.jsonl`，保留每行的两个 judge、阈值拒绝原因和是否进入训练；正式实验应报告保留率和 judge disagreement，而不是只提交筛选后的 pair。

## 统一实验输出

```bash
PYTHONPATH=src uv run scripts/run_experiment.py data/raw/scenarios.jsonl \
  --output-dir artifacts/experiment \
  --methods base,prompt_only,sft,direct_dpo,pc_dpo \
  --backend dry_run --samples 3 --scorer none
```

正式运行时用 `--model-revision <Hub commit hash>` 固定权重；本地 LoRA checkpoint 的 base model revision 也会写入训练配置和实验 manifest。

正式 HF 对比必须为 `sft,direct_dpo,pc_dpo` 分别提供 `--model-map sft=... ,direct_dpo=...,pc_dpo=...`；程序会拒绝把同一个 Base checkpoint 冒充训练方法。所有方法使用同一推理模板，避免测试时提示差异混入结果。

每次运行会创建 `artifacts/experiment/<run_id>/`：

```text
experiment_manifest.json       # 输入哈希、方法、split、variants、seed
scores.jsonl                 # 全部方法的逐样本结果
<method>/run_manifest.json   # 模型、命令、seed、source commit
<method>/predictions.jsonl   # 统一预测记录
<method>/scores.jsonl        # 逐样本评分
report/report.md             # 组会/论文初读
report/report.html           # 浏览器报告
report/report_manifest.json  # scores 输入哈希和 rubric 版本
report/summary.csv           # Excel/R/脚本读取
report/summary.json          # 机器可读聚合
report/review_queue.jsonl    # 低置信度和规则冲突样本
validation.json              # 服务器/CI 完整性检查结果
```

正式报告按 method 和 `method/family` 汇总 Trait Fidelity、Behavior Validity、Truthfulness、Safety 和 Sycophancy。关键词规则只允许通过 `score_outputs.py --smoke-only` 做管道 smoke check，绝不能作为论文指标；论文主结果应使用独立 `llm_score_outputs.py`，并固定 judge checkpoint、temperature=0、rubric 版本。`run_experiment.py --scorer none` 只生成预测，`--scorer llm` 需要认证的 judge endpoint。先打开 `report.html`，再抽查：

```bash
PERSONA_RL_REVIEWER=alice PYTHONPATH=src uv run scripts/review_results.py \
  artifacts/experiment/<run_id>/report/review_queue.jsonl \
  --output artifacts/experiment/<run_id>/report/human_audit.jsonl --limit 100
PYTHONPATH=src uv run scripts/summarize_audit.py \
  artifacts/experiment/<run_id>/report/human_audit.jsonl \
  --output artifacts/experiment/<run_id>/report/audit_summary.json
```

人工审计只记录 accept/reject/uncertain、1–5 质量分和备注，不参与训练或调参。外部效度评分使用盲法程序：

```bash
PERSONA_RL_REVIEWER=alice PYTHONPATH=src uv run scripts/human_score.py \
  artifacts/experiment/<run_id>/predictions.jsonl \
  --output artifacts/experiment/<run_id>/report/human_scores.jsonl --limit 120
```

该程序随机打乱回答、用不可逆 item code 隐藏 method，并分别询问 trait fidelity、hidden behavior validity、truthfulness、safety、confidence 五个维度；结果不回流训练。

低人力设置下先从完整分数生成固定审计队列，再交互式阅读，避免把所有 sampling 当作独立人工样本：

```bash
PYTHONPATH=src uv run scripts/make_audit_queue.py \
  artifacts/experiment/<run_id>/scores.jsonl \
  --output artifacts/experiment/<run_id>/report/audit_queue.jsonl --fraction 0.2 --seed 7
PERSONA_RL_REVIEWER=alice PYTHONPATH=src uv run scripts/review_results.py \
  artifacts/experiment/<run_id>/report/audit_queue.jsonl \
  --output artifacts/experiment/<run_id>/report/human_audit.jsonl --limit 100
```

默认 `run_experiment.py` 对每个样本生成 `canonical,paraphrase,minimal,formal,terse,conversational` 六种 prompt/style variant，用于把 trait code 与表面语气正交化。`summary.json` 额外写出 ICC(2,1)、配对组数量、可用指标的 bootstrap 95% CI；只有生成时提供 `--capability-count`，`capability_retention` 才会显示为数值，否则报告为 `NA`。配对差异可进一步导出：

```bash
PYTHONPATH=src uv run scripts/compare_methods.py \
  artifacts/experiment/<run_id>/scores.jsonl \
  --output artifacts/experiment/<run_id>/report/pairwise.csv
PYTHONPATH=src uv run scripts/validate_artifacts.py \
  artifacts/experiment/<run_id> \
  --expected-methods base,prompt_only,sft,direct_dpo,pc_dpo \
  --expected-variants canonical,paraphrase,minimal
```

推理多 GPU 时让每个 rank 写独立 shard，避免并发写同一个 JSONL：

```bash
PYTHONPATH=src torchrun --nproc_per_node=4 scripts/run_inference.py data/raw/scenarios.jsonl \
  --method pc_dpo --model artifacts/checkpoints/pc_dpo \
  --output artifacts/predictions.jsonl --split test --samples 3 \
  --variants canonical,paraphrase,minimal
PYTHONPATH=src uv run scripts/merge_predictions.py artifacts \
  --pattern 'predictions.rank*.jsonl' --output artifacts/predictions.merged.jsonl
```

公开资源下载：

```bash
uv run scripts/fetch_public.py https://example.org/file.jsonl data/raw/external.jsonl --sha256 <known_sha256>
```

## 公开 baseline 与统一输出

源码由 GitHub CLI 获取并固定 commit：Machine-Mindset、PersLLM、Geometry-of-Personality、RoleLLM、PersonaGym、Representation Engineering、BIG5-CHAT 和 SimPO。完整论文简介与 BibTeX 见 [`REFERENCES.md`](REFERENCES.md)。不要复制论文中的数字；先运行：

```bash
PYTHONPATH=src uv run scripts/clone_baselines.py --root . --manifest configs/baseline_manifest.json
```

原仓库命令可以只做计划（默认），或在对应依赖准备好后执行：

```bash
PYTHONPATH=src uv run scripts/run_official_baseline.py persllm --root .
PYTHONPATH=src uv run scripts/run_official_baseline.py personality_vector --root .
PYTHONPATH=src uv run scripts/run_official_baseline.py personagym --root . --execute
```

也可以使用统一 wrapper。它保留原仓库命令，不重写算法；给出 raw CSV/JSONL 后会自动调用统一 schema 转换器：

```bash
PYTHONPATH=src uv run scripts/run_baseline.py machine_mindset --root .
PYTHONPATH=src uv run scripts/run_baseline.py simpo --root .
PYTHONPATH=src uv run scripts/run_baseline.py machine_mindset --root . \
  --raw-output artifacts/raw_machine.csv \
  --scenarios data/raw/scenarios.jsonl \
  --normalized-output artifacts/official_predictions.jsonl \
  --model-override FarReelAILab/Machine_Mindset_en_ISTJ
```

归一化结果具有与内部方法相同的 `PredictionRecord` 字段，之后复用 `score_outputs.py`、`llm_score_outputs.py`、`render_report.py` 和 `validate_artifacts.py`。`RoleLLM`、`PersonaGym`、`BIG5-CHAT` 属于 benchmark/data-only 或 dataset/model source，wrapper 会明确记录这一限制，不会生成伪造的训练结果。

原方法输出格式不同，统一前必须转换为以下字段：`scenario_id,response,method,model_id,temperature`。转换命令：

```bash
PYTHONPATH=src uv run scripts/normalize_external.py official.csv data/raw/scenarios.jsonl \
  --output artifacts/official_predictions.jsonl \
  --source-manifest artifacts/official_runs/machine_mindset/manifest.json \
  --method-override machine_mindset \
  --model-override FarReelAILab/Machine_Mindset_en_ISTJ
PYTHONPATH=src uv run scripts/score_outputs.py artifacts/official_predictions.jsonl \
  data/raw/scenarios.jsonl --output artifacts/official_scores.jsonl
PYTHONPATH=src uv run scripts/llm_score_outputs.py artifacts/official_predictions.jsonl \
  data/raw/scenarios.jsonl --output artifacts/official_llm_scores.jsonl \
  --model Qwen/Qwen2.5-72B-Instruct --second-model gpt-4o-mini
PYTHONPATH=src uv run scripts/render_report.py artifacts/official_scores.jsonl \
  --output-dir artifacts/official_report
```

`external/` 中的原代码保持原样；`run_official_baseline.py` 只负责 provenance 和命令执行，`normalize_external.py` 只负责 schema 转换。主论文的同条件比较应报告 source repo、commit、模型 revision、训练 token、数据 split 和命令日志，避免自建 baseline 的选择性实现。

`run_experiment.py` 默认拒绝把外部方法伪装成 prompt proxy；只有临时 smoke 才可显式传 `--allow-proxy-external`。正式外部基线必须经过原仓库命令、原始输出日志和 `normalize_external.py`，然后再进入同一个 scorer/report。

## 训练

单卡：

```bash
PYTHONPATH=src uv run scripts/train_dpo.py data/processed/judged_pairs.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct --method direct-dpo \
  --output artifacts/checkpoints/direct_dpo --seed 7
PYTHONPATH=src uv run scripts/train_sft.py data/processed/judged_pairs.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output artifacts/checkpoints/sft --seed 7
```

多卡使用 Accelerate/torchrun。先执行 `uv run accelerate config`，再：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
  accelerate launch --num_processes 4 scripts/train_dpo.py \
  data/processed/judged_pairs.jsonl --method pc-dpo \
  --output artifacts/checkpoints/pc_dpo_seed7 --seed 7
```

训练入口使用锁定环境中的 TRL `DPOTrainer`/`SFTTrainer`，默认 PEFT LoRA + 4-bit QLoRA；`--no-qlora` 仅用于小模型 CPU smoke。PC-DPO 的多头 reward 在 `judge_pairs.py` 中独立缓存，`train_dpo.py --method pc-dpo` 将 criterion/invariance/truth/safety/uncertainty 组成约束权重，对训练 pair 做确定性过滤与 0–2 次重加权；它与 Direct-DPO 使用不同的训练目标。

`src/persona_rl/constraints.py` 给出可复现标量目标；`src/persona_rl/pc_dpo_trainer.py` 将 DPO loss、criterion/invariance/truth/safety/uncertainty penalty 实际接入 `compute_loss`，对每个 batch 反向传播。默认入口采用 offline constraint-weighted PC-DPO；论文不得把它描述为在线 PPO。

正式运行建议：每个 seed 独立输出目录；启用 `--gradient_checkpointing`、bf16 和定期 checkpoint；保存 `config.json`、模型 revision、git commit、训练日志和环境导出。断点恢复使用 TRL 的 `resume_from_checkpoint`，不要覆盖已完成 seed。

## 评测与统计

评测器应输出一行一个样本：

```json
{"id":"s_000001","method":"pc_dpo","trait_fidelity":0.8,"behavior_validity":1.0,"truthfulness":1.0,"safety":1.0,"sycophancy":0.0,"seed":7}
```

聚合：

```bash
PYTHONPATH=src uv run scripts/evaluate.py artifacts/raw_eval.jsonl --output-path artifacts/metrics.json
```

主统计使用同一情境/视图/seed 的配对 bootstrap；同一模型的多次 sampling 不能当成独立模型。训练和测试 judge 必须分离，至少 20% 测试集用第二 judge 复核。

## 资源估算

| 阶段 | 推荐资源 | 估算 |
|---|---|---|
| 场景生成 | CPU 4 核或 API 调用 | 1–4 小时 |
| 自动 judge | API 或 1 张 24 GB GPU | 10^4–10^5 次调用 |
| 7B QLoRA 单 seed | 1–3 张 24 GB GPU | 0.5–2 天 |
| 3 seeds + 迁移模型 | 4 张 24 GB GPU | 2–5 天 |
| 完整评测 | CPU/GPU 混合 | 1–3 天 |
| 短程长对话 | 5,000–40,000 次推理 | 0.5–3 天 |

实际成本取决于输出长度、judge 模型和是否使用 API。所有 API 结果必须缓存；不要重复调用相同的 `(model, prompt, seed, rubric_version)`。

## 服务器检查清单

```bash
python --version
nvidia-smi
uv --version
uv sync --extra train
PYTHONPATH=src uv run scripts/generate_scenarios.py --count 20 --output /tmp/smoke.jsonl
PYTHONPATH=src uv run scripts/build_preferences.py /tmp/smoke.jsonl --output-path /tmp/smoke_pref.jsonl
python -m compileall src scripts
```

首次正式训练前先完成两周 pilot：确认 Direct-DPO 存在代理错位、PC-DPO 能提高隐藏行为效度、第二 judge 不推翻主结论，然后再扩展到 3 seeds 和第二模型。
