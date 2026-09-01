# 单服务器 Smoke 与模型分层

本文记录当前共享节点的实测环境和模型选择。Smoke 只验证工程链路，不产生论文结论。

## 实测环境

当前节点探测到 8 张 NVIDIA GeForce RTX 3090，每张 24GB 显存；已有 Python 3.10 的 `personaforge_env`，包含 PyTorch 2.10、Transformers 5.2、Datasets、Accelerate、PEFT 和 ModelScope。节点磁盘余量约 8GB，GPU 是共享资源，运行前必须重新检查占用，不能杀掉其他用户进程。

已实测：

- ModelScope `Qwen/Qwen2.5-0.5B-Instruct`：约 942MB 权重，单 GPU 生成 18 条预测，成功产出 scores/report/trait-style artifacts；适合接口和 schema smoke，不适合评价人格能力。
- 已有缓存 `Meta-Llama-3-8B-Instruct`：约 15GB 权重，单 GPU 生成 18 条预测成功；适合作为加载和推理压力测试，不代表主实验结果。

## 推荐分层

| 层级 | policy 模型 | 单卡建议 | 用途 |
|---|---|---:|---|
| L0 | Qwen2.5-0.5B-Instruct | 8–12GB | 数据、推理、报告、adapter 冒烟 |
| L1 | Qwen2.5-1.5B-Instruct | 16–24GB | 首轮真实训练/消融 sanity check |
| L2 | Qwen2.5-3B-Instruct | 24GB（4-bit） | 小规模行为 pilot，磁盘足够时使用 |
| L3 | Qwen2.5-7B-Instruct | 24GB（QLoRA） | 论文主模型，3 seeds |
| L4 | Llama-3.1-8B-Instruct 或其他异源 7B–9B | 24GB（QLoRA） | 跨模型家族迁移 |

结论：下一轮快速训练选 **1.5B**；0.5B 只用于 CI/smoke，不能据此判断 CB-DPO 是否有效；正式论文仍使用 7B–8B。32B–72B 模型只作为独立 generator/judge，通过 vLLM 或 API 运行，不和 policy 争抢单卡。

## 在节点上运行

```bash
export REPO=/data1/$USER/Persona_RL/persona_rl
export PYTHON=/data1/$USER/miniconda3/envs/personaforge_env/bin/python
export PYTHONPATH=$REPO/src
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=/data1/$USER/.cache/modelscope
cd "$REPO"

# 先确认 GPU 没有被占用；GPU 编号是宿主机编号
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv

# 下载 L1 模型。若磁盘不足，不要删除他人文件；先换到有空间的挂载点。
$PYTHON scripts/download_model.py Qwen/Qwen2.5-1.5B-Instruct \
  --output /data1/$USER/models/qwen25-1.5b --source modelscope

$PYTHON scripts/generate_scenarios.py \
  --output data/raw/server_smoke_1_5b.jsonl --count 60 \
  --capability-count 6 --seed 41 --split-strategy family_holdout --languages zh,en

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/run_experiment.py \
  data/raw/server_smoke_1_5b.jsonl --output-dir artifacts/server_smoke_1_5b \
  --methods base --model-map base=/data1/$USER/models/qwen25-1.5b \
  --backend hf --split test --samples 1 --temperature 0.2 \
  --max-new-tokens 32 --variants canonical,terse --scorer none

RUN=$(find artifacts/server_smoke_1_5b -name predictions.jsonl | sort | tail -1 | sed 's#/predictions.jsonl##')
$PYTHON scripts/score_outputs.py "$RUN/predictions.jsonl" \
  data/raw/server_smoke_1_5b.jsonl --output "$RUN/smoke_scores.jsonl" --smoke-only
$PYTHON scripts/render_report.py "$RUN/smoke_scores.jsonl" --output-dir "$RUN/report"
$PYTHON scripts/audit_dataset.py data/raw/server_smoke_1_5b.jsonl --output "$RUN/report/data_audit.json"
```

如果要做 1.5B 的训练 smoke，先用 `build_preferences.py` 生成少量占位 pair，仅检查 TRL/LoRA 是否能保存 checkpoint；占位 pair 不得用于论文训练：

```bash
$PYTHON scripts/build_preferences.py data/raw/server_smoke_1_5b.jsonl \
  --output-path /tmp/smoke_preferences.jsonl
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train_dpo.py /tmp/smoke_preferences.jsonl \
  --model /data1/$USER/models/qwen25-1.5b --method direct-dpo \
  --output /tmp/persona_rl_train_smoke --epochs 0.01 \
  --batch-size 1 --grad-accumulation 1 --max-length 512 \
  --no-qlora --no-bf16
```

服务器的 Python 3.10 兼容回退已经在 `results.py`、`inference.py` 和 `run_experiment.py` 中实现。若训练环境没有 TRL，命令应明确失败并提示安装 `uv sync --extra train`；不要从其他用户的 Conda 环境直接复制包。

## 产物判定

Smoke 通过的最低条件是：模型加载成功、预测行数等于 `test × samples × variants`、PredictionRecord 的语言与 Scenario 一致、无重复 `(method, scenario, variant, sample)`、报告和数据审计文件存在。Smoke 分数只能用于检查管道，不得写入论文主表。
