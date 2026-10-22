"""
make_tables.py
==============

Writes the numeric tables of the report as LaTeX fragments, directly from the
verified outputs of verify_and_analyze.py, so that no table value is typed by
hand.

Usage:  python make_tables.py      (run verify_and_analyze.py first)
Output: ../Report_Source/tables/*.tex
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TAB = HERE.parent / "Report_Source" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

NAME = {"Original PPO": "1 Baseline", "Time Penalty": "2 Time penalty",
        "Hover Reward": "3 Hover reward", "Corrected Reward": "4 Corrected reward"}
MINUS = "−"


def f(x, d=1):
    s = f"{x:,.{d}f}"
    return s.replace("-", MINUS)


def p(x):
    return "< 0.001" if x < 0.001 else f"{x:.2f}" if x >= 0.01 else f"{x:.3f}"


def write(name, header, rows, spec):
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("wrote", name)


s = pd.read_csv(OUT / "per_agent_statistics.csv").set_index("agent")
pc = pd.read_csv(OUT / "paired_comparisons_verified.csv")
order = list(NAME)

# Table: score statistics
rows = []
for a in order:
    r = s.loc[a]
    rows.append([NAME[a], f(r.mean_score, 2),
                 f"[{f(r.mean_score_boot95_low)}, {f(r.mean_score_boot95_high)}]",
                 f(r.sd_score, 1), f(r.median_score, 1),
                 f"{f(r.q1_score)} to {f(r.q3_score)}", f(r.min_score), f(r.max_score),
                 str(int(r.episodes_score_ge_200))])
write("tab_scores", ["Agent", "Mean", "95\\% CI of mean", "SD", "Median", "IQR", "Min", "Max", "$\\geq$200"],
      rows, "lrcrrcrrr")

# Table: landing outcomes and duration
rows = []
for a in order:
    r = s.loc[a]
    rows.append([NAME[a], f"{int(r.successes)}/100",
                 f"[{f(r.success_wilson95_low_pct)}, {f(r.success_wilson95_high_pct)}]",
                 str(int(r.terminated_without_landing)), str(int(r.time_limit_episodes)),
                 f(r.mean_duration, 1), f"[{f(r.mean_duration_t95_low, 0)}, {f(r.mean_duration_t95_high, 0)}]",
                 f(r.median_duration, 1)])
write("tab_outcomes", ["Agent", "Landed", "95\\% Wilson CI (\\%)", "Crash/OOB", "Time limit",
                       "Mean dur.", "95\\% CI", "Median dur."], rows, "lrcrrrcr")

# Table: conditional statistics
rows = []
for a in order:
    r = s.loc[a]
    na = "n/a"
    rows.append([NAME[a],
                 na if pd.isna(r.mean_score_successful) else f(r.mean_score_successful, 1),
                 f(r.mean_score_unsuccessful, 1),
                 na if pd.isna(r.mean_duration_successful) else f(r.mean_duration_successful, 0),
                 f(r.mean_duration_unsuccessful, 0)])
write("tab_conditional", ["Agent", "Score, landed", "Score, not landed", "Duration, landed",
                          "Duration, not landed"], rows, "lrrrr")

# Table: slow/upright metrics
bins = pd.read_csv(OUT / "slow_upright_fraction_bins.csv").set_index("model")
rows = []
for a in order:
    r = s.loc[a]
    rows.append([NAME[a], f(r.mean_slow_upright_frames, 2), f(r.sd_slow_upright_frames, 1),
                 f(r.median_slow_upright_frames, 1), f(r.mean_per_episode_slow_fraction_pct, 1),
                 f(r.pooled_slow_fraction_pct, 1), str(int(bins.loc[a, ">=75%"]))])
write("tab_slow", ["Agent", "Mean frames", "SD", "Median", "Mean share (\\%)", "Pooled share (\\%)",
                   "Episodes $\\geq$75\\%"], rows, "lrrrrrr")

# Table: paired comparisons
short = {"Original PPO": "Baseline", "Time Penalty": "Time pen.", "Hover Reward": "Hover",
         "Corrected Reward": "Corrected"}
rows = []
for _, r in pc.iterrows():
    b, a = r.comparison.split(" minus ")
    rows.append([f"{short[b]} $-$ {short[a]}", f(r.mean_score_diff, 1),
                 f"[{f(r.score_diff_boot95_low)}, {f(r.score_diff_boot95_high)}]",
                 p(r.wilcoxon_p_holm),
                 f"{f(r.success_diff_pp, 0)}", f"[{f(r.success_diff_boot95_low_pp, 0)}, {f(r.success_diff_boot95_high_pp, 0)}]",
                 f"{int(r.discordant_b_only)} / {int(r.discordant_a_only)}", p(r.mcnemar_p_holm)])
write("tab_paired", ["Comparison", "$\\Delta$ score", "95\\% CI", "$p_{W}$", "$\\Delta$ landed (pp)",
                     "95\\% CI", "Discordant", "$p_{M}$"], rows, "lrcrrccr")

# Table: integrity checks (from verification log)
log = (OUT / "verification_log.txt").read_text().splitlines()
rows = []
for line in log:
    line = line.strip()
    if line.startswith("[PASS]") or line.startswith("[FAIL]"):
        rows.append([line[7:].strip()[:1].upper() + line[7:].strip()[1:].replace("<=", "$\\leq$").replace("scores > 100", "scores above 100"), line[1:5]])
write("tab_integrity", ["Check performed on final\\_results.csv", "Result"], rows, "lc")

# Table: cross-check of the preliminary summary in the project brief
cc = pd.read_csv(OUT / "cross_check_against_brief.csv").set_index("agent")
rows = []
for a in order:
    r = cc.loc[a]
    rows.append([NAME[a], f(r.brief_mean_score, 2), f(s.loc[a].mean_score, 3),
                 f"{int(r.brief_success_pct)}\\%", f"{int(s.loc[a].successes)}/100",
                 f(r.brief_slow_frames, 2), f(s.loc[a].mean_slow_upright_frames, 2),
                 "Match" if r.match else "Mismatch"])
write("tab_crosscheck", ["Agent", "Brief score", "Recomputed", "Brief landing", "Recomputed",
                         "Brief frames", "Recomputed", "Status"], rows, "lrrrrrrc")
