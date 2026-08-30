# Persona-RL Literature and Baseline References

This file is the citation inventory for the experiments. Each source is classified by what can be compared fairly: a trainable method, a released checkpoint, a representation-control method, or an evaluation resource. Report the pinned repository commit and model revision alongside every external result.

## Closely related work: PersonaForge

Tong and Zou (Findings of ACL 2026) propose a psychology-grounded three-layer personality architecture and a selective dual-process generation mechanism with an Inner Monologue workspace. The paper reports orthogonality tests for Big Five plus defense-mechanism dimensions, long-dialogue drift, RoleBench transfer, and ablations of the dual-process component. It is the closest conceptual reference for our motivation, definitions, long-dialogue analysis, and discussion of psychology-grounded constraints.

**Classification in this project:** related work and conceptual reference only. PersonaForge is not registered in `baseline_manifest.json`, is not accepted by the unified baseline runner, and must not appear as a row in the main or appendix method-comparison tables. Its published results may be discussed qualitatively, but they must not be merged with results produced by this experiment pipeline.

Paper: https://aclanthology.org/2026.findings-acl.386/  
PDF: https://aclanthology.org/2026.findings-acl.386.pdf  
Official repository: https://github.com/fQwQf/PersonaForge

```bibtex
@inproceedings{tong2026personaforge,
  title={PersonaForge: Psychology-Grounded Dual-Process Architecture for Personality-Consistent Role-Playing Agents},
  author={Tong, Jizhou and Zou, Sirui},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={7845--7874},
  year={2026}
}
```

## Core personality measurement and evaluation

### Personality Traits in Large Language Models

Serapio-Garcia et al. study whether language models express Big Five traits and provide the conceptual starting point for treating personality as a measurable behavioral construct rather than a prompt label. Cite for the motivation and measurement framing, not as a training baseline.

```bibtex
@article{serapio2023personality,
  title={Personality Traits in Large Language Models},
  author={Serapio-Garcia, Greg and Safdari, Mustafa and Crepy, Clement and Sun, Luning and Fitz, Stephen and Romero, Peter and Abdulhai, Marwa and Faust, Aleksandra and Mataric, Maja J.},
  journal={arXiv preprint arXiv:2307.00184},
  year={2023}
}
```

Paper: https://arxiv.org/abs/2307.00184

### PersonaLLM

Jiang et al. evaluate whether LLMs can express personality traits under controlled prompts. It is useful as an evaluation precedent and for separating trait expression from general helpfulness.

Paper: https://doi.org/10.18653/v1/2024.findings-naacl.229

### PsychoBench

Huang et al. introduce a psychological portrayal benchmark and questionnaire-based evaluation. The repository is included under `data/raw/PsychoBench` as frozen evaluation data; it must not be mixed into training.

Paper: https://arxiv.org/abs/2310.01386  
Repository: https://github.com/thu-coai/PsychoBench

```bibtex
@inproceedings{huang2024humanity,
  title={Who is ChatGPT? Benchmarking LLMs' Psychological Portrayal Using PsychoBench},
  author={Huang, Jen-tse and Wang, Wenxuan and Li, Eric John and Lam, Man Ho and Ren, Shujie and Yuan, Youliang and Jiao, Wenxiang and Tu, Zhaopeng and Lyu, Michael R.},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

### PersonaGym

Samuel et al. evaluate persona agents in dynamic environments. PersonaGym is an external evaluation suite, not a training algorithm. Use it for transfer evaluation or report it as benchmark-only.

Paper: https://arxiv.org/abs/2407.18416  
Repository: https://github.com/vsamuel2003/PersonaGym

```bibtex
@article{samuel2024personagym,
  title={PersonaGym: Evaluating Persona Agents and LLMs},
  author={Samuel, Vinay and Zou, Henry Peng and Zhou, Yue and Chaudhari, Shreyas and Kalyan, Ashwin and Rajpurohit, Tanmay and Deshpande, Ameet and Narasimhan, Karthik and Murahari, Vishvak},
  journal={arXiv preprint arXiv:2407.18416},
  year={2024}
}
```

## Trainable personality and role baselines

### Machine Mindset

Cui et al. train MBTI-specific models with multi-stage pretraining, instruction tuning, and DPO-style preference optimization. It is the strongest released-checkpoint comparison for direct categorical personality alignment, but its MBTI labels and base models differ from our Big Five-like target vectors.

Paper: https://arxiv.org/abs/2312.12999  
Repository: https://github.com/PKU-YuanGroup/Machine-Mindset

```bibtex
@misc{cui2023machine,
  title={Machine Mindset: An MBTI Exploration of Large Language Models},
  author={Cui, Jiaxi and Lv, Liuzhenghao and Wen, Jing and Wang, Rongsheng and Tang, Jing and Tian, Yonghong and Yuan, Li},
  year={2023},
  eprint={2312.12999},
  archivePrefix={arXiv}
}
```

### PersLLM

Zeng et al. propose personified training with automatically generated data, consistency objectives, and DPO-style preference data. This is the primary external direct-personality training baseline. The public checkout requires a ModelCenter-format checkpoint and its original Harry Potter data, so a strict reproduction must retain those prerequisites in the run manifest.

Paper: https://arxiv.org/abs/2407.12393  
Repository: https://github.com/Ellenzzn/PersLLM

```bibtex
@article{zeng2024persllm,
  title={PersLLM: A Personified Training Approach for Large Language Models},
  author={Zeng, Zheni and Chen, Jiayi and Chen, Huimin and Yan, Yukun and Chen, Yuxuan and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2407.12393},
  year={2024}
}
```

### BIG5-CHAT

Li et al. shape LLM personalities using human-grounded Big Five data. This is a useful data/checkpoint baseline because its trait vocabulary is close to psychometrics, but the release is primarily a dataset and model source rather than a drop-in trainer for our schema.

Paper: https://arxiv.org/abs/2410.16491  
Repository: https://github.com/wenkai-li/Big5-Chat  
Dataset: https://huggingface.co/datasets/wenkai-li/big5_chat

```bibtex
@inproceedings{li2025big5chat,
  title={BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data},
  author={Li, Wenkai and Liu, Jing and Liu, Andy and Zhou, Xuhui and Diab, Mona and Sap, Maarten},
  booktitle={Proceedings of ACL},
  year={2025}
}
```

### RoleLLM

Wang et al. construct role profiles, generate role-conditioned instructions, and train RoleLLaMA/RoleGLM with RoCIT. RoleLLM is a role-playing rather than psychometric-trait method, making it a useful adjacent supervised-training baseline. The public repository contains RoleBench assets and documentation but no complete local trainer.

Paper: https://arxiv.org/abs/2310.00746  
Repository: https://github.com/InteractiveNLP-Team/RoleLLM-public  
Dataset: https://huggingface.co/datasets/ZenMoore/RoleBench

```bibtex
@article{wang2023rolellm,
  title={RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models},
  author={Wang, Zekun Moore and Peng, Zhongyuan and Que, Haoran and Liu, Jiaheng and Zhou, Wangchunshu and Wu, Yuhan and Guo, Hongcheng and Gan, Ruitong and Ni, Zehao and Yang, Jian and others},
  journal={arXiv preprint arXiv:2310.00746},
  year={2023}
}
```

## Representation and preference-optimization baselines

### The Geometry of Personality

The public toolkit extracts contrastive Big Five activation vectors from Gemma-2-2B-IT and applies forward hooks during generation. It is the closest non-training personality-control baseline. Because the repository is notebook-centered and uses cached vector files, report layer, vector source, steering strength, and cache hash.

Paper/toolkit: https://github.com/gunmayhanda/The-Geometry-of-Personality  
Repository commit used here: `47628a9982c4532d19b16d7aa4c66228d0f304ad`

### Representation Engineering (RepE)

Zou et al. introduce representation reading and control for high-level concepts such as honesty, safety, and emotions. RepE is not specifically a personality trainer, but it is an important parameter-free or low-parameter control comparison and provides the official `rep-reading`/`rep-control` pipelines.

Paper: https://arxiv.org/abs/2310.01405  
Repository: https://github.com/andyzoujm/representation-engineering

```bibtex
@misc{zou2023transparency,
  title={Representation Engineering: A Top-Down Approach to AI Transparency},
  author={Zou, Andy and Phan, Long and Chen, Sarah and Campbell, James and Guo, Phillip and Ren, Richard and Pan, Alexander and Yin, Xuwang and Mazeika, Mantas and others},
  year={2023},
  eprint={2310.01405},
  archivePrefix={arXiv}
}
```

### SimPO

Meng et al. propose reference-free preference optimization with an average log-probability reward and a target margin. It is a general preference-optimization baseline, not a personality method. Include it to test whether improvements come from the psychometric constraint rather than from using a different preference objective.

Paper: https://arxiv.org/abs/2405.14734  
Repository: https://github.com/princeton-nlp/SimPO  
Pinned commit: `1b3e8f3528a23bce3da514a2dce8ea7490d4bc75`

```bibtex
@inproceedings{meng2024simpo,
  title={SimPO: Simple Preference Optimization with a Reference-Free Reward},
  author={Meng, Yu and Xia, Mengzhou and Chen, Danqi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## How to cite our comparisons

Use the primary paper citation for scientific claims and the repository URL plus pinned commit for implementation provenance. Never copy reported benchmark numbers from an external README into the Persona-RL result table. Re-run the official command when possible, normalize its raw output into `PredictionRecord`, and state explicitly when a method is `released_checkpoint`, `official_training`, `official_notebooks`, `dataset_only`, or `benchmark_only`.
