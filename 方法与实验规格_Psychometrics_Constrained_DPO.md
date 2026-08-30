# Psychometrics-Constrained DPO：方法与实验规格

## 0. 论文主张

本文只主张：直接人格偏好优化可能提高问卷或风格分数，却不一定提高人格相关的隐藏行为；加入反事实选择性、跨风格不变性和真实性/安全约束后，可以得到更稳定、行为有效且安全的条件化行为策略。论文使用“条件化行为倾向”，不声称模型具有人类人格。

**方法名称与实现边界：** 本文方法是 **offline constraint-weighted PC-DPO (CB-DPO)**。`judge_pairs.py` 产生独立多头 reward，`constraints.py` 定义约束权重和标量损失，`train_dpo.py --method pc-dpo` 使用该权重做确定性样本重加权。除非另行实现并报告在线 reward-model trainer，不得把本实现描述为 PPO、在线 RL 或“完全可学习的多头 reward model”。

## 1. 研究假设

**H1 代理投机：** Direct-DPO 在原始人格 rubric 上升分，但在不含人格词的隐藏行为任务上提升小或不提升。

**H2 效度修复：** 加入隐藏行为效度约束后，人格分数与行为结果的相关性提高。

**H3 不变性：** 加入多视图约束后，提示改写、题序、温度和语言变化下的 ICC 提高。

**H4 安全性：** 直接优化宜人性、外向性或开放性可能增加 sycophancy、幻觉或越权；真实性/安全硬约束可以抑制回归。

预注册成功标准：主方法相对 Direct-DPO 的隐藏行为效度差值 bootstrap 95% CI 不跨零；ICC 提高；安全/真实性不显著恶化；通用能力下降不超过 3 个百分点。仅问卷分数提升不算成功。

## 2. 统一实验基础

### 2.1 模型

主模型固定一个公开 7B–8B instruct checkpoint（例如 Qwen2.5-7B-Instruct 或 Llama-3.1-8B-Instruct），迁移验证用不同家族的 7B–8B instruct 模型。使用同一 tokenizer、LoRA rank 16、最大长度、解码参数和训练 token 预算。pilot 用 1 个 seed，正式结果用 3 个 seed；每个测试条件 3 次采样、temperature 0.2/0.7。

### 2.2 人格目标

主论文只做三个维度，每个维度取低、中、高三档（-1, 0, +1）：

- 尽责性：计划、检查、约束遵循；
- 宜人性：尊重表达、温和纠正、合作；
- 诚实-谦逊：承认不确定性、不夸大能力、利益冲突中不占便宜。

禁止把宜人性定义为“永远同意”。每个维度都要有目标行为和禁止替代：

| 维度 | 目标行为 | 禁止替代 |
|---|---|---|
| 尽责性 | 分解步骤、检查约束、报告未完成事项 | 过度冗长、僵化拒绝 |
| 宜人性 | 尊重表达、温和纠正、合作解决 | 无条件附和、隐藏分歧 |
| 诚实-谦逊 | 承认局限、披露利益冲突 | 自我贬低、过度拒答 |

### 2.3 情境数据

建立 18 个情境族，每个维度 6 个。每个反事实组只改变一个 trait（其余两个固定为 0），并携带至少两个结构化行为选项和 `gold_option`；同一情境的正负 trait 不得复用同一个 gold label。当前生成器已实现这一约束，测试必须检查每个组的 Hamming distance 恰为 1。

#### 数据来源与许可

数据不要声称全部来自“人类人格数据”。建议采用四层来源并在 release 中逐层标记 provenance：

1. **量表层：** 使用 IPIP 公版题目；如使用 BFI-2/NEO，必须核对许可，不要直接重新分发受版权保护的完整题目。量表只用于 secondary evaluation，不直接作为训练答案。
2. **情境层：** 由研究者编写 18 个情境族的行为定义、硬约束和判定规则；这是最重要、也是必须人工设计的少量内容，不需要大规模人工标注。
3. **扩写层：** 使用与被测模型不同家族的开放模型生成角色、主题、措辞和对话历史；每条扩写保留模板 ID、生成模型、版本和随机种子。
4. **标签层：** 由程序规则、可执行结果和两个不同 judge 生成偏好/评分。judge 只作为 noisy annotator，不能被描述为 ground truth；分歧样本进入争议集。

外部数据（PersonaGym、PsychoBench、InCharacter 等）只作为 frozen holdout 或 baseline 评测，必须遵守各自 license，不应把其测试答案混入训练集。

#### 顶会级质量要求

LLM 合成数据本身通常不足以支撑顶会主张。需要补上以下证据：

- **规则可判定性：** 至少一半主终点是结构化选择、约束检查或可执行环境结果，而非开放式主观打分；
- **跨 judge 一致性：** 两个不同模型 judge 的 Krippendorff alpha 或 pairwise agreement 达到预设阈值；低于阈值的样本不训练；
- **反事实完整性：** 每条关键样本都有同一情境/不同 trait 和同一 trait/不同情境的配对 ID；
- **模板外推：** 测试按情境族和主题留出，证明不是记忆模板；
- **污染检查：** 对公共 benchmark、训练语料和模型输出做 n-gram/embedding 相似度检查，并发布过滤脚本；
- **小规模外部审计：** 即使不做大规模人工标注，也建议对每类情境随机抽查 20–50 条，合计约 100–300 条，由 1–2 位研究者检查 rubric 是否可判定。此审计不参与训练和调参，只用于报告数据质量上限；
- **版本化与可追溯：** 发布模板、生成 prompt、模型 hash、随机种子、规则代码、原始候选和过滤日志。

因此，这个数据集更适合作为**方法论文的可复现实验资源**，而不是单独宣称“人类心理学金标准数据集”。如果没有上述小规模审计，论文应把结论限定为“在自动构造情境上的方法比较”。

每条样本至少包含：

~~~json
{
  "family": "deadline_check",
  "target_z": {"conscientiousness": 1, "agreeableness": 0, "honesty_humility": 1},
  "situation": "...",
  "hard_constraints": ["不得声称未执行的检查已经完成"],
  "behavior_rubric": ["提出最小可行检查步骤", "说明剩余不确定性"],
  "forbidden_behaviors": ["无条件同意", "伪造完成状态"],
  "hidden_task": {"type": "binary_decision", "gold_behavior": 1},
  "counterfactual_group": "cf_0042"
}
~~~

建议训练/验证/测试为 6,000/1,000/2,000 个情境。测试按情境族、主题和人格组合划分，而非随机句子划分。测试增加同义改写、人格词屏蔽、中文/英文和无关上下文版本。

### 2.4 自动标签

每个情境由 base model、prompt baseline 和独立生成模型生成 4–8 个候选。先做硬约束过滤，再由两个不同模型 judge 按隐藏行为、trait、真实性、安全、criterion、invariance 和 uncertainty 打分；只保留高置信、非 tie、双 judge 同意的 pair。`llm_score_outputs.py --second-model` 对最终测试输出做双 judge 数值聚合；人工程序 `human_score.py` 只做盲法外部效度抽查。任何关键词计数都只能用于 smoke test，不能进入论文指标。

### 2.5 奖励头

冻结或训练四个独立 head：

- r_trait：目标人格行为匹配；
- r_criterion：隐藏行为效度；
- r_truth：真实性和不确定性校准；
- r_safe：安全、隐私和越权。

额外计算 sycophancy 惩罚 r_syc 和 judge 不确定性 u。优化分数：

$$R=r_{trait}+\lambda_c r_{criterion}+\lambda_i r_{invariance}-\lambda_u u.$$

硬约束为 $r_{truth}\geq\tau_t$、$r_{safe}\geq\tau_s$、$KL(\pi_\theta\|\pi_{ref})\leq\epsilon$。

## 3. 三个核心模型

1. **Base：** 原始 instruct 模型；
2. **Direct-DPO：** 只使用人格 rubric 构造偏好对，标准 DPO；
3. **PC-DPO/CB-DPO：** 使用行为效度、不变性、不确定性和硬约束构造偏好对，并按约束权重进行离线重加权。

所有模型使用相同数据量、LoRA 配置、训练步数和验证集搜索预算。Direct-DPO 必须是认真复现的强 baseline，不能故意削弱。

PC-DPO 的离线训练目标：

$$L=L_{DPO}+\lambda_c L_{criterion}+\lambda_i L_{invariance}+\lambda_u L_{uncertainty}+\lambda_s L_{safety}+\lambda_{KL}L_{KL}.$$

实现上先对每个情境的 6 个提示/风格视图生成候选并缓存 reward。代码中的标量目标为：

$$\ell_{pc}=w(r)\ell_{dpo}+0.25[(1-r_{criterion})+(1-r_{invariance})]+0.25u,$$

其中 $w(r)=0$ 对安全/真实性失败样本，其他样本被裁剪到 $[0.05,2]$。`pc_dpo_trainer.py` 将该目标实际接入 TRL 的 `compute_loss`，对 DPO margin、约束 penalty 和相对参考模型的 KL 近似项共同反向传播；`training_config.json` 记录 objective、constraint heads 和训练配置。

## 4. 必做实验

### E1 原始人格量表

使用 IPIP/BFI 项目作为 secondary metric。每个模型和目标向量使用随机题序、正反向项目、3 次采样和两个 temperature，报告均值、方差和 CI。量表分数不能作为唯一成功标准。

### E2 隐藏行为效度（主终点）

测试不出现人格词，包含：

- 尽责性：是否执行必要检查、承认未完成事项；
- 宜人性：能否礼貌纠正错误主张；
- 诚实-谦逊：利益冲突中是否披露并选择公平方案；
- 综合任务：计划执行、隐私边界、错误纠正和风险决策。

用 LLM judge 对结构化选项/动作结果进行盲法评分，并报告 judge 间 ICC、与人工盲审的相关性、行为准确率、AUROC、Brier score、calibration error，以及人格分数与行为结果的相关性。关键词匹配不得作为主指标。主张必须建立在 E2，而不是 E1。

### E3 提示不变性

对每个测试情境生成：语序改写、同义词改写、去除人格词、加入无关上下文、中文/英文版本。报告 trait score ICC、行为选择一致率、奖励方差和目标档位分类准确率。

### E4 维度纠缠

固定情境，只改变一个 trait code，计算：

$$A_{jk}=E[Y_j|do(z_k=+1)]-E[Y_j|do(z_k=-1)].$$

报告对角 target effect、非对角 leakage 和稀疏度。重点检查宜人性是否增加 sycophancy、尽责性是否造成过度冗长、诚实-谦逊是否造成过度拒答。

### E5 安全和能力

测试错误事实附和、越权、隐私泄露、危险请求、prompt injection、事实校准、数学和代码。报告绝对分数及相对 base 的变化。

### E6 短程长对话

用 100 个自动 episode、每个 30 轮，不依赖真人：

- 1–10 轮中性任务；
- 11–15 轮人格任务；
- 16–20 轮错误记忆和冲突提示；
- 21–25 轮跨主题迁移；
- 26–30 轮安全/真实性压力。

报告轮级 ICC、矛盾率、冲突后恢复轮数和记忆污染率。E6 是 robustness extension，不是唯一主证据。

## 5. 基线与消融

基线：prompt-only、LoRA/SFT、Personality Vector/activation steering、Direct-DPO、KTO（可选）、PC-DPO。

必须消融：

1. 去掉 L_criterion：检测问卷投机；
2. 去掉 L_invariance：检测稳定性贡献；
3. 去掉 uncertainty penalty：检测奖励过度自信；
4. 将安全硬约束改为普通加权奖励；
5. 使用同源 judge 训练和测试；
6. 训练数据量 25%、50%、100%；
7. 第二模型家族迁移。

## 6. 统计方案

配对单位是同一个情境、目标向量、提示视图和采样种子。主检验使用配对 bootstrap 10,000 次和 95% CI；按人格维度、情境族和模型家族分层。补充报告 Cliff's delta、ICC CI、sycophancy 的 Wilson CI、judge 一致率和等效性检验。

同一模型的多次 sampling 不能当成独立模型。不能用测试集选择 lambda、checkpoint 或提示模板。

## 7. 两周 pilot

**Day 1–2：** 300 个情境、schema、规则、两个 judge、完整指标脚本；先跑 Base，确认指标有方差。

**Day 3–5：** 500–1,000 个样本训练 Direct-DPO，跑 E1–E5，寻找“问卷提升但隐藏行为不提升”。

**Day 6–9：** 加入 criterion/invariance/hard constraint，训练三组权重，和 Direct-DPO 配对比较。

**Day 10–12：** 做人格词屏蔽、第二 judge、新情境族、错误事实和 temperature 攻击。

**Day 13–14：** 只有在以下条件同时满足时才扩展到 3 seeds 和第二模型：Direct-DPO 存在可复现的代理错位；PC-DPO 提高隐藏行为；第二 judge 不推翻结论；安全/真实性无明显回归；成本可扩展。

## 8. 顶会主表和关键图

| 方法 | Trait Fidelity | 隐藏行为效度 | ICC | 矛盾率 | sycophancy | 安全违反 | 能力变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base |  |  |  |  |  |  |  |
| Prompt-only |  |  |  |  |  |  |  |
| LoRA/SFT |  |  |  |  |  |  |  |
| Personality Vector |  |  |  |  |  |  |  |
| Direct-DPO |  |  |  |  |  |  |  |
| PC-DPO |  |  |  |  |  |  |  |

必须画：问卷分数–隐藏行为效度散点图；提示改写/语言/温度–ICC 曲线；人格强度–sycophancy/真实性/能力曲线。

## 9. 复现交付物

开源情境模板、生成脚本、split ID、judge prompt 和缓存、训练配置、adapter/checkpoint hash、原始评测 JSON、统计脚本和失败案例。不要收集真实用户人格数据；本方案不需要大规模人工偏好标注。

## 10. 最终选题边界

第一篇论文只保留一个主方法：PC-DPO。因果解耦、完整 Pareto frontier 和长期动态只作为小规模扩展。论文必须回答：

1. Direct-DPO 是否系统性优化了人格代理而非行为？
2. PC-DPO 是否在未见情境和第二 judge 上修复该错位？
3. 修复是否以真实性、安全性和能力为代价？

若不能同时回答，缩小论文主张，不要宣称模型“拥有了人格”。
