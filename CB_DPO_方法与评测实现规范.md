# CB-DPO 方法与评测实现规范

## 论文中可以声称什么

本项目研究的是条件化行为倾向，不是模型是否具有人的内在人格。可证伪主张是：在相同底座、数据预算和推理模板下，Direct-DPO 可能提高人格 rubric/问卷分数，却不能稳定改变未见情境中的结构化行为；offline constraint-weighted CB-DPO 能提高隐藏行为效度、跨风格不变性和 trait selectivity，同时不牺牲真实性、安全性与通用能力。

## 方法

每个训练样本有 `(situation, target_z, options, gold_option)`。反事实组只改变一个 trait，其他维度为 0；因此可估计 `do(z_k=+1)-do(z_k=-1)`，而不是把多维变化误当人格效应。

候选回答由独立生成模型产生。两个与被测模型不同的 judge 分别输出：

- `trait_score`：目标行为是否符合 trait code；
- `criterion_score`：是否完成情境中隐藏的行为标准；
- `invariance_score`：四种表达风格下行为选择是否一致；
- `truth_score`、`safety_score`：硬约束；
- `uncertainty`：judge 对该判定的不确定度。

只保留两个 judge 都选择同一 winner、非 tie、truth/safety 达标且 uncertainty 足够低的 pair。`constraints.py` 的目标是：

```text
w(r) = 0                                      if truth < .7 or safety < .7
w(r) = clip((trait + criterion + invariance)/3 * (1 - uncertainty), .05, 2)
ell = w(r) * ell_DPO + .25*(1-criterion) + .25*(1-invariance) + .25*uncertainty + .05*KL(policy || reference)
```

工程实现同时执行两层约束：训练数据阶段按权重 0、1、2 做确定性过滤/重加权；`pc_dpo_trainer.py` 在每个 batch 的 `compute_loss` 中对 DPO margin 叠加 criterion、invariance、truth、safety 和 uncertainty penalty，并反向传播。它不是在线 PPO，准确名称是 offline constraint-weighted PC-DPO。

## 评测优先级

1. **主终点：隐藏行为效度。** 去掉人格词，要求模型在结构化选项、计划约束、利益冲突和错误纠正任务中选择可判定动作。由独立 LLM judge 盲评；报告行为准确率、AUROC、Brier、校准误差和 judge 间 ICC。
2. **表面依赖检验。** 对同一情境做六种 style，使用 `style_normalize.py` 统一为中性专业表达并要求保持动作/事实不变，再评测；比较原始输出与 style-normalized 输出的行为效度差。
3. **因果选择性。** 每组只改变一个 trait，报告目标 trait effect、非目标 leakage、target-to-leakage ratio。
4. **安全与能力。** 错误事实、越权、隐私、危险请求、数学、代码和事实校准；报告绝对值及相对 Base 的变化。
5. **长对话。** 只在首轮设置 target，随后跨主题和冲突压力；报告轮级一致性、恢复轮数和矛盾率。

关键词匹配只能验证管道连通性，禁止进入论文指标。正式评分使用 `llm_score_outputs.py --second-model ...`；外部效度使用 `human_score.py` 的盲法 1–5 分程序，100–200 个分层样本即可，不参与训练或调参。风格中介分析使用 `style_normalize.py` 生成独立缓存，再对归一化文件重新执行 LLM judge。

## 必须保留的消融

- Direct-DPO（相同候选与预算，仅 trait preference）；
- 去掉 criterion、invariance、uncertainty、hard safety constraint；
- 同源 judge 与异源 judge；
- style-normalization 前后；
- 一个 trait 与三个 trait 同时改变；
- 新 family、跨语言、第二模型家族。

只有当 Direct-DPO 在原始 rubric 上升、但在隐藏行为/风格归一化后不升，而 CB-DPO 在第二 judge 和盲法人审上仍保持优势时，才能声称发现并修复了表面人格投机。
