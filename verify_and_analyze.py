"""
verify_and_analyze.py
=====================

Independent verification and statistical analysis of the LunarLander-v3 PPO
reward-shaping evaluation dataset (final_results.csv).

This script is NEW analysis code written for the final report. It does not
train or run any reinforcement-learning agent; it only reads the 400 recorded
evaluation episodes and recomputes every statistic quoted in the report.

Usage
-----
    python verify_and_analyze.py [path/to/final_results.csv]

Default input : ../Original_Supplied_Materials/final_results.csv
Outputs       : ./outputs/*.csv, ./outputs/*.json, ./outputs/verification_log.txt

Dependencies  : Python >= 3.10, pandas, numpy, scipy
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "Original_Supplied_Materials" / "final_results.csv"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

ORDER = ["Original PPO", "Time Penalty", "Hover Reward", "Corrected Reward"]
TIME_LIMIT = 1000           # LunarLander-v3 max_episode_steps
N_BOOT = 10_000             # bootstrap resamples
RNG_SEED = 20261021         # fixed so every interval is reproducible
LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def t_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.5 + conf / 2, len(x) - 1) * se
    return m - h, m + h


def boot_ci(x: np.ndarray, fn, rng: np.random.Generator, conf: float = 0.95):
    """Percentile bootstrap CI of statistic fn over resampled episodes."""
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    reps = fn(x[idx], axis=1)
    a = (1 - conf) / 2
    return float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ---------------------------------------------------------------------------
# 1. Load and verify integrity
# ---------------------------------------------------------------------------
def load_and_verify(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    log(f"Input file: {path.name}")
    log(f"Rows: {len(df)}  Columns: {list(df.columns)}")
    expected_cols = ["model", "episode", "seed", "score", "duration_frames",
                     "successful_landing", "slow_upright_airborne_frames"]
    checks = {
        "exact column set": list(df.columns) == expected_cols,
        "400 rows": len(df) == 400,
        "no missing values": not df.isna().any().any(),
        "four expected model labels": set(df["model"]) == set(ORDER),
        "100 episodes per model": all(df["model"].value_counts() == 100),
        "seeds 1000-1099 for every model": all(
            sorted(g["seed"]) == list(range(1000, 1100)) for _, g in df.groupby("model")),
        "seed = 999 + episode": bool((df["seed"] - df["episode"] == 999).all()),
        "success flag is binary": set(df["successful_landing"].unique()) <= {0, 1},
        "duration within 1..1000": bool(df["duration_frames"].between(1, TIME_LIMIT).all()),
        "slow/upright frames <= duration": bool(
            (df["slow_upright_airborne_frames"] <= df["duration_frames"]).all()),
        "no success at the 1000-frame limit": not bool(
            ((df["duration_frames"] == TIME_LIMIT) & (df["successful_landing"] == 1)).any()),
        "every success scores > 100": bool(
            (df.loc[df["successful_landing"] == 1, "score"] > 100).all()),
    }
    for name, ok in checks.items():
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit("Integrity check failed - see log above.")
    # The Hover agent's training bonus could add up to ~1000 points per episode.
    # If evaluation scores had included it, scores would be strongly positive.
    hv_max = df.loc[df["model"] == "Hover Reward", "score"].max()
    log(f"  [INFO] Max Hover-Reward evaluation score = {hv_max:.3f} "
        "(consistent with scoring under the unmodified environment reward)")
    return df


# ---------------------------------------------------------------------------
# 2. Per-agent descriptive statistics
# ---------------------------------------------------------------------------
def classify(row) -> str:
    if row["successful_landing"] == 1:
        return "Landed (came to rest)"
    if row["duration_frames"] >= TIME_LIMIT:
        return "Time limit reached"
    return "Terminated without landing"


def per_agent(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for m in ORDER:
        g = df[df["model"] == m]
        s = g["score"].to_numpy()
        d = g["duration_frames"].to_numpy()
        h = g["slow_upright_airborne_frames"].to_numpy()
        k = int(g["successful_landing"].sum())
        n = len(g)
        wl, wh = wilson_ci(k, n)
        tl, th = t_ci(s)
        bml, bmh = boot_ci(s, np.mean, rng)
        bmdl, bmdh = boot_ci(s, np.median, rng)
        dl, dh = t_ci(d.astype(float))
        succ = g[g["successful_landing"] == 1]
        fail = g[g["successful_landing"] == 0]
        rows.append({
            "agent": m,
            "episodes": n,
            "mean_score": s.mean(),
            "sd_score": s.std(ddof=1),
            "sem_score": s.std(ddof=1) / np.sqrt(n),
            "mean_score_t95_low": tl, "mean_score_t95_high": th,
            "mean_score_boot95_low": bml, "mean_score_boot95_high": bmh,
            "median_score": float(np.median(s)),
            "median_score_boot95_low": bmdl, "median_score_boot95_high": bmdh,
            "q1_score": float(np.quantile(s, .25)), "q3_score": float(np.quantile(s, .75)),
            "min_score": s.min(), "max_score": s.max(),
            "episodes_score_ge_200": int((s >= 200).sum()),
            "successes": k,
            "success_rate_pct": 100 * k / n,
            "success_wilson95_low_pct": 100 * wl, "success_wilson95_high_pct": 100 * wh,
            "time_limit_episodes": int((d >= TIME_LIMIT).sum()),
            "terminated_without_landing": int(((d < TIME_LIMIT) & (g["successful_landing"] == 0)).sum()),
            "mean_duration": d.mean(), "sd_duration": d.std(ddof=1),
            "mean_duration_t95_low": dl, "mean_duration_t95_high": dh,
            "median_duration": float(np.median(d)),
            "mean_duration_successful": succ["duration_frames"].mean() if len(succ) else np.nan,
            "mean_duration_unsuccessful": fail["duration_frames"].mean(),
            "mean_score_successful": succ["score"].mean() if len(succ) else np.nan,
            "mean_score_unsuccessful": fail["score"].mean(),
            "mean_slow_upright_frames": h.mean(),
            "sd_slow_upright_frames": h.std(ddof=1),
            "median_slow_upright_frames": float(np.median(h)),
            "mean_per_episode_slow_fraction_pct": 100 * np.mean(h / d),
            "pooled_slow_fraction_pct": 100 * h.sum() / d.sum(),
            "total_frames": int(d.sum()),
            "total_slow_upright_frames": int(h.sum()),
            "spearman_duration_vs_score": float(stats.spearmanr(d, s).statistic),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Seed-paired comparisons
# ---------------------------------------------------------------------------
def paired(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    wide = {m: df[df["model"] == m].set_index("seed").sort_index() for m in ORDER}
    rows = []
    for a, b in combinations(ORDER, 2):
        A, B = wide[b], wide[a]          # difference reported as b - a
        diff = (A["score"] - B["score"]).to_numpy()
        lo, hi = boot_ci(diff, np.mean, rng)
        sa, sb = A["successful_landing"].to_numpy(), B["successful_landing"].to_numpy()
        n10 = int(((sa == 1) & (sb == 0)).sum())   # b succeeds, a fails
        n01 = int(((sa == 0) & (sb == 1)).sum())   # a succeeds, b fails
        sdiff = (sa - sb).astype(float)
        slo, shi = boot_ci(sdiff, np.mean, rng)
        hdiff = (A["slow_upright_airborne_frames"] - B["slow_upright_airborne_frames"]).to_numpy()
        hlo, hhi = boot_ci(hdiff, np.mean, rng)
        ddiff = (A["duration_frames"] - B["duration_frames"]).to_numpy()
        dlo, dhi = boot_ci(ddiff, np.mean, rng)
        rows.append({
            "comparison": f"{b} minus {a}",
            "mean_score_diff": diff.mean(),
            "score_diff_boot95_low": lo, "score_diff_boot95_high": hi,
            "median_score_diff": float(np.median(diff)),
            "paired_cohens_dz": diff.mean() / diff.std(ddof=1),
            "wilcoxon_p": float(stats.wilcoxon(diff).pvalue),
            "success_diff_pp": 100 * sdiff.mean(),
            "success_diff_boot95_low_pp": 100 * slo, "success_diff_boot95_high_pp": 100 * shi,
            "discordant_b_only": n10, "discordant_a_only": n01,
            "mcnemar_exact_p": mcnemar_exact(n10, n01),
            "slow_frames_diff": hdiff.mean(),
            "slow_frames_diff_boot95_low": hlo, "slow_frames_diff_boot95_high": hhi,
            "duration_diff": ddiff.mean(),
            "duration_diff_boot95_low": dlo, "duration_diff_boot95_high": dhi,
            "spearman_seed_score": float(stats.spearmanr(A["score"], B["score"]).statistic),
        })
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"].tolist())
    out["mcnemar_p_holm"] = holm(out["mcnemar_exact_p"].tolist())
    return out


def outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(outcome=df.apply(classify, axis=1))
    tab = pd.crosstab(df["model"], df["outcome"]).reindex(ORDER)
    cols = ["Landed (came to rest)", "Terminated without landing", "Time limit reached"]
    for c in cols:
        if c not in tab:
            tab[c] = 0
    return tab[cols]


def slow_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(frac=df["slow_upright_airborne_frames"] / df["duration_frames"])
    bins = [0, .10, .25, .50, .75, 1.0001]
    labels = ["<10%", "10-25%", "25-50%", "50-75%", ">=75%"]
    df["bin"] = pd.cut(df["frac"], bins=bins, labels=labels, right=False)
    return pd.crosstab(df["model"], df["bin"]).reindex(ORDER)


# ---------------------------------------------------------------------------
# 4. Cross-check against the summary supplied in the original package/brief
# ---------------------------------------------------------------------------
BRIEF = {  # values quoted in the project brief
    "Original PPO": (177.53, 75, 28.14),
    "Time Penalty": (-14.98, 27, 148.82),
    "Hover Reward": (-60.63, 0, 842.38),
    "Corrected Reward": (90.21, 55, 261.74),
}


def cross_check(summary: pd.DataFrame, supplied: Path | None) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        bs, bk, bh = BRIEF[r["agent"]]
        rows.append({
            "agent": r["agent"],
            "brief_mean_score": bs, "recomputed_mean_score": round(r["mean_score"], 2),
            "brief_success_pct": bk, "recomputed_success_pct": r["success_rate_pct"],
            "brief_slow_frames": bh, "recomputed_slow_frames": round(r["mean_slow_upright_frames"], 2),
            "match": (abs(round(r["mean_score"], 2) - bs) < 0.005
                      and r["success_rate_pct"] == bk
                      and abs(round(r["mean_slow_upright_frames"], 2) - bh) < 0.005),
        })
    cc = pd.DataFrame(rows)
    log("\nCross-check against the summary quoted in the project brief:")
    for _, r in cc.iterrows():
        log(f"  [{'MATCH' if r['match'] else 'MISMATCH'}] {r['agent']}: "
            f"score {r['recomputed_mean_score']:.2f} vs {r['brief_mean_score']:.2f}; "
            f"success {r['recomputed_success_pct']:.0f}% vs {r['brief_success_pct']}%; "
            f"slow frames {r['recomputed_slow_frames']:.2f} vs {r['brief_slow_frames']:.2f}")
    if supplied is not None and supplied.exists():
        sup = pd.read_csv(supplied).set_index("model")
        log("\nCross-check against supplied summary_statistics.csv:")
        pairs = [("mean_score", "mean_score"), ("score_sd", "sd_score"),
                 ("median_score", "median_score"), ("mean_duration_frames", "mean_duration"),
                 ("median_duration_frames", "median_duration"),
                 ("mean_fraction_slow_upright_pct", "mean_per_episode_slow_fraction_pct"),
                 ("time_limit_1000_frames", "time_limit_episodes"),
                 ("episodes_score_ge_200", "episodes_score_ge_200")]
        s2 = summary.set_index("agent")
        for a in ORDER:
            bad = [c for c, mine in pairs if abs(sup.loc[a, c] - s2.loc[a, mine]) > 1e-6]
            log(f"  [{'MATCH' if not bad else 'MISMATCH: ' + ', '.join(bad)}] {a}")
    return cc


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    rng = np.random.default_rng(RNG_SEED)
    log("=" * 72)
    log("LunarLander-v3 PPO reward-shaping study: data verification log")
    log("=" * 72)
    df = load_and_verify(path)

    summary = per_agent(df, rng)
    pairs = paired(df, rng)
    outcomes = outcome_table(df)
    bins = slow_bins(df)
    cc = cross_check(summary, path.parent / "summary_statistics.csv")

    summary.to_csv(OUT / "per_agent_statistics.csv", index=False)
    pairs.to_csv(OUT / "paired_comparisons_verified.csv", index=False)
    outcomes.to_csv(OUT / "episode_outcomes.csv")
    bins.to_csv(OUT / "slow_upright_fraction_bins.csv")
    cc.to_csv(OUT / "cross_check_against_brief.csv", index=False)
    df.assign(outcome=df.apply(classify, axis=1),
              slow_upright_fraction=df["slow_upright_airborne_frames"] / df["duration_frames"]
              ).to_csv(OUT / "episodes_with_derived_columns.csv", index=False)

    log("\nPer-agent statistics (rounded):")
    show = summary.set_index("agent")[[
        "mean_score", "sd_score", "median_score", "successes", "success_wilson95_low_pct",
        "success_wilson95_high_pct", "mean_duration", "median_duration",
        "mean_slow_upright_frames", "mean_per_episode_slow_fraction_pct", "pooled_slow_fraction_pct"]]
    log(show.round(2).T.to_string())
    log("\nEpisode outcomes:")
    log(outcomes.to_string())
    log("\nSeed-paired comparisons (difference = second minus first):")
    log(pairs[["comparison", "mean_score_diff", "score_diff_boot95_low", "score_diff_boot95_high",
               "wilcoxon_p_holm", "success_diff_pp", "mcnemar_p_holm"]].round(4).to_string(index=False))

    meta = {
        "bootstrap_resamples": N_BOOT, "rng_seed": RNG_SEED, "time_limit_frames": TIME_LIMIT,
        "confidence_level": 0.95,
        "notes": [
            "Intervals describe evaluation-episode variability for one trained checkpoint per "
            "reward condition; they are not intervals over independent training runs.",
            "Outcome classes are inferred: success flag = 1 -> landed; duration = 1000 -> time "
            "limit; otherwise terminated without landing (crash or out-of-bounds; not separable).",
        ],
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(meta, indent=2))
    (OUT / "verification_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
