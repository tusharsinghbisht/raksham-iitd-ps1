"""EER with speaker-clustered bootstrap CIs, tau frozen on dev, test rates at tau.

tau is chosen on the dev split as the call-level score where genuine false alarms = 1%.
Test detection/false-alarm rates are then read at that frozen tau (Wilson 95% CIs).
"""
import argparse
import json

import numpy as np
import pandas as pd

TARGET_FPR = 0.01


def eer(gen, fake):
    s = np.concatenate([gen, fake])
    y = np.concatenate([np.zeros(len(gen)), np.ones(len(fake))])
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    tpr = np.cumsum(y) / max(1, y.sum())
    fpr = np.cumsum(1 - y) / max(1, (1 - y).sum())
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2)


def boot_eer(df, col, n=500, seed=0):
    rng = np.random.default_rng(seed)
    g = {k: v[col].to_numpy() for k, v in df[df.label == "genuine"].groupby("speaker")}
    f = {k: v[col].to_numpy() for k, v in df[df.label == "fake"].groupby("speaker")}
    gk, fk = list(g), list(f)
    out = []
    for _ in range(n):
        gs = np.concatenate([g[k] for k in rng.choice(gk, len(gk))])
        fs = np.concatenate([f[k] for k in rng.choice(fk, len(fk))])
        out.append(eer(gs, fs))
    return np.percentile(out, [2.5, 97.5]).tolist()


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0, c - h), min(1, c + h)]


def pct(x):
    return f"{100 * x:.1f}%"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--out", default="results/results")
    a = ap.parse_args()
    df = pd.read_csv(a.scores)
    hi = df[df.source != "team"]
    team = df[df.source == "team"]
    res, lines = {}, []
    lines.append("| Condition | EER whole clip | EER call-level (median 4 s) [95% CI] | tau (dev, 1% FA) | Test detection @tau | Test false alarm @tau | Team false alarm @tau |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in hi.condition.unique():
        d = hi[hi.condition == c]
        dv, te = d[d.split == "dev"], d[d.split == "test"]
        e_full = eer(te[te.label == "genuine"].z_full.values, te[te.label == "fake"].z_full.values)
        e_call = eer(te[te.label == "genuine"].z_call.values, te[te.label == "fake"].z_call.values)
        ci = boot_eer(te, "z_call")
        tau = float(np.quantile(dv[dv.label == "genuine"].z_call.values, 1 - TARGET_FPR))
        tg, tf = te[te.label == "genuine"].z_call.values, te[te.label == "fake"].z_call.values
        det = wilson(int((tf >= tau).sum()), len(tf))
        fa = wilson(int((tg >= tau).sum()), len(tg))
        tm = team[team.condition == c].z_call.values
        tfa = wilson(int((tm >= tau).sum()), len(tm))
        per_gen = {g: wilson(int((v.z_call.values >= tau).sum()), len(v))[0]
                   for g, v in te[te.label == "fake"].groupby("generator")}
        res[c] = dict(eer_full=e_full, eer_call=e_call, eer_call_ci=ci, tau=tau, test_detection=det, test_false_alarm=fa,
                      team_false_alarm=tfa, detection_by_generator=per_gen,
                      n_test_genuine=len(tg), n_test_fake=len(tf), n_team=len(tm),
                      n_speakers_test=int(te.speaker.nunique()))
        lines.append(f"| {c} | {pct(e_full)} | {pct(e_call)} [{pct(ci[0])}–{pct(ci[1])}] | {tau:.2f} | "
                     f"{pct(det[0])} ({len(tf)}) | {pct(fa[0])} ({len(tg)}) | {pct(tfa[0])} ({len(tm)}) |")
    shared = set(hi[hi.label == "genuine"].speaker.astype(str)) & set(hi[hi.label == "fake"].speaker.astype(str))
    lines.append("")
    lines.append(f"Same-person check (real speaker X vs clone of X), {len(shared)} speakers, all splits, call-level EER:")
    for c in hi.condition.unique():
        d = hi[(hi.condition == c) & hi.speaker.astype(str).isin(shared)]
        e = eer(d[d.label == "genuine"].z_call.values, d[d.label == "fake"].z_call.values)
        res[c]["eer_same_person"] = e
        lines.append(f"- {c}: {pct(e)} ({(d.label == 'genuine').sum()} real, {(d.label == 'fake').sum()} cloned)")
    lines.append("")
    lines.append("Detection by generator @tau (test): ")
    for c, r in res.items():
        lines.append(f"- {c}: " + ", ".join(f"{g} {pct(v)}" for g, v in r["detection_by_generator"].items()))
    open(a.out + ".json", "w").write(json.dumps(res, indent=1))
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
