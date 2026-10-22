# Reinforcement Learning for Autonomous Lunar Landing
### Reward Shaping, Policy Optimization, and Reward Hacking

**Raja Halazon** · Independent engineering project · October 2026

This repository documents a controlled experiment on reward design in reinforcement learning. Four Proximal Policy Optimization (PPO) agents were trained in Gymnasium's `LunarLander-v3`. They are identical except for the reward used during training. All four were then evaluated on the same 100 starting conditions under the **original, unmodified** environment reward. One of the four rewards was deliberately flawed so that its failure, a form of reward hacking, could be produced, measured, and corrected.

📄 **Full report:** [`Final Engineering Report.pdf`](Final%20Engineering%20Report.pdf) (editable version: [`.docx`](Final%20Engineering%20Report.docx))

---

## Motivation

In reinforcement learning, the reward function *is* the task specification. An agent optimizes the reward it is given, not the intent behind it. When a reward pays for something that only resembles progress, a capable optimizer can learn to collect that payment instead of finishing the task. This is known as specification gaming or reward hacking. This project reproduces that failure in a small, fully observable setting and tests a correction.

## Software

| Component | Used in the experiment |
|---|---|
| Language | Python 3.12 |
| Environment | Gymnasium `LunarLander-v3` (Box2D), discrete actions |
| RL algorithm | Stable-Baselines3 PPO, `MlpPolicy` (library-default hyperparameters assumed) |
| Backend | PyTorch |
| Reward changes | Custom Gymnasium reward wrappers (training only) |

Exact package versions were not recorded. The analysis code in this repository needs only `pandas`, `numpy`, `scipy`, and `matplotlib` (see `Analysis/requirements.txt`).

## Experimental method

- **Training:** one PPO run per reward condition, about 500,000 environment steps each, CPU, training seeds not fixed.
- **Evaluation:** 100 episodes per agent with environment seeds **1000 to 1099** (the same seeds for every agent), deterministic actions, wrappers removed. Every score is therefore the **unmodified** LunarLander reward.
- **Recorded per episode:** score, duration (steps), landing flag (episode ended with the lander at rest), and the number of slow/upright airborne frames (|vx| < 0.2, |vy| < 0.2, |θ| < 0.15 rad, both legs off the ground).

## The four reward functions

Each agent was trained on `r̃ = r + F`, where `r` is the base environment reward.

| Agent | Added term `F` per step | Purpose |
|---|---|---|
| 1 Baseline | 0 | Reference performance |
| 2 Time penalty | −0.01 | Test whether penalizing duration improves efficiency |
| 3 Hover reward | +1 if \|vx\| < 0.2, \|vy\| < 0.2, \|θ\| < 0.15 and the episode has not ended | Deliberately flawed incentive, used to study reward hacking |
| 4 Corrected reward | 2·[S(sₜ₊₁) − S(sₜ)] − 0.01, with S(s) = −(\|vx\| + \|vy\| + \|θ\|) | Reward *improving* stability rather than *holding* it |

The corrected term telescopes over an episode to `2·[S(s_T) − S(s_0)] − 0.01·T`. Holding any fixed state therefore earns nothing from it, which removes the incentive Agent 3 can exploit.

## Verified key results

All values below were recomputed from the raw file `final_results.csv` (400 episodes). 95% confidence intervals describe variation across the 100 evaluation starts for each trained checkpoint, **not** across independent training runs.

| Agent | Mean score [95% CI] | Median | Landed (of 100) [Wilson 95% CI] | Reached 1,000-step limit | Mean duration (steps) | Slow/upright airborne share |
|---|---:|---:|---:|---:|---:|---:|
| 1 Baseline | **177.5** [156.9, 197.0] | 227.7 | **75** [65.7%, 82.5%] | 1 | 301.9 | 8.9% |
| 2 Time penalty | −15.0 [−40.3, 11.3] | −73.4 | 27 [19.3%, 36.4%] | 1 | 382.4 | 30.7% |
| 3 Hover reward | −60.6 [−67.9, −54.2] | −53.9 | **0** [0.0%, 3.7%] | **87** | 931.9 | **86.6%** |
| 4 Corrected reward | 90.2 [65.3, 114.5] | 150.3 | 55 [45.2%, 64.4%] | 9 | 503.8 | 43.3% |

Seed-paired comparisons (same starting conditions):

- **Corrected vs. hover:** +150.8 points [126.6, 174.4]; +55 percentage points in landings (55 seeds favor the corrected agent, 0 favor the hover agent).
- **Corrected vs. baseline:** −87.3 points [−113.9, −61.9]; −20 percentage points in landings (Holm-adjusted McNemar p = 0.001).

![Landing rate and episode outcomes](Figures/Fig04_landing_success_and_outcomes.png)

![Distribution of episode scores](Figures/Fig05_score_distributions.png)

![Slow, upright, airborne behavior](Figures/Fig07_slow_upright_behavior.png)

![Reward-hacking mechanism (conceptual diagram)](Figures/Fig09_reward_hacking_mechanism.png)

## Conclusions

1. **Baseline.** Unmodified PPO came to rest in 75/100 episodes and had the best score on every task metric.
2. **Time penalty.** The −0.01 per-step agent landed in 27/100 episodes and took *longer*, not shorter. The penalty is worth at most about 1 point of discounted return, and each reward was trained only once, so this gap cannot be attributed to the penalty. It does **not** show that time penalties are harmful in general.
3. **Reward hacking.** The hover-bonus agent never landed, ran to the time limit in 87/100 episodes, and spent 86.6% of its time in the rewarded state. The bonus can be collected on every frame, while a landing pays once and ends the episode. The evidence strongly supports reward hacking as the explanation; a causal demonstration would require repeated training and trajectory logs.
4. **Correction.** Rewarding the *change* in stability recovered 55/100 landings but did not reach the baseline. The correction mitigated this failure. It did not prove that reward hacking was eliminated, and it did not produce an optimal reward.

## Repository structure

```
Final Engineering Report.pdf / .docx   Final report (PDF and editable Word version)
README.md                              This file
Figures/                               11 figures, 300-dpi PNG and vector PDF
Analysis/                              New analysis code written for the final report
  verify_and_analyze.py                Integrity checks and every statistic in the report
  make_figures.py                      Regenerates all figures
  make_tables.py                       Writes the report's numeric tables from verified outputs
  requirements.txt
  outputs/                             Verified results (CSV/JSON) and the verification log
Report_Source/                         LaTeX source of the report and the Word build script
Original_Supplied_Materials/           Original project files, unchanged
  final_results.csv                    Primary experimental record (400 evaluation episodes)
  summary_statistics.csv, paired_comparisons.csv, data_validation.json,
  analyze_results.py, README.md, figures/, Lunar_Lander_Engineering_Project_Report.pdf
```

`final_results.csv` is the only primary experimental data. The other files in `Original_Supplied_Materials/` came from an earlier analysis of that file. They are kept for traceability, and the new analysis reproduces every value in them.

## Reproducing the analysis

```bash
cd Analysis
python -m pip install -r requirements.txt
python verify_and_analyze.py   # integrity checks + all statistics -> outputs/
python make_figures.py         # all figures -> ../Figures/
python make_tables.py          # LaTeX tables -> ../Report_Source/tables/
```

Bootstrap intervals use 10,000 resamples and the fixed seed 20261021, so a re-run reproduces every number exactly. Figures use the Inter typeface if it is installed (optionally set `FIG_FONT_DIR` to a folder of Inter `.ttf` files) and otherwise fall back to the matplotlib default font.

To rebuild the report, compile `Report_Source/report.tex` twice with XeLaTeX, then run `python build_docx.py` (requires pandoc 3 and python-docx; LibreOffice is optional and is used only to pre-fill the Word table of contents).

## What cannot be reproduced yet

The training scripts (`train*.py`), reward-wrapper source, trained checkpoints (`lunar_lander_*.zip`), training logs, and package versions were not part of the evidence package. They are therefore **not** included here, and no substitute training code is provided. The evaluation protocol (seeds 1000–1099, deterministic actions, unmodified reward) is specified precisely enough to re-run once the checkpoints are restored.

## Limitations and next steps

- One training run per reward condition: differences between agents mix the reward effect with run-to-run variance. **Next step:** 5–10 recorded seeds per condition, reported with interquartile means and stratified bootstrap intervals.
- The slow/upright counter is not proof of stationary hovering. **Next step:** log position, velocity, actions, and terminal cause per step, and record short videos.
- Single coefficients (−0.01, +1, 2) were tested with default hyperparameters and a 500k-step budget. **Next step:** coefficient sweeps, learning curves for both the training reward and the original reward, and a strictly potential-based variant `2·[γ·S(s') − S(s)]`.

## References (selected)

1. J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
2. Farama Foundation, [Gymnasium: Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/).
3. A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *JMLR* 22(268), 2021; [PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
4. A. Y. Ng, D. Harada, S. Russell, "Policy Invariance under Reward Transformations," *ICML*, 1999.
5. J. Skalse et al., "Defining and Characterizing Reward Gaming," *NeurIPS* 35, 2022.
6. V. Krakovna et al., "[Specification gaming: the flip side of AI ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)," DeepMind, 2020.
7. R. Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice," *NeurIPS* 34, 2021.

The complete IEEE-style reference list is in the report.
