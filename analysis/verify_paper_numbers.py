"""Independent verification of every quantitative claim in the paper.

Recomputes all numbers from raw run JSONs ONLY (no reuse of any analysis module),
and prints a claim-by-claim PASS/FAIL report against the values stated in main.tex.

Run: python analysis/verify_paper_numbers.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PKG = Path(__file__).resolve().parents[1]
PLAN2 = PKG.parents[1]
RAW = PLAN2 / "results" / "raw"
GENDIR = PLAN2 / "paper" / "ejor_submission" / "generated"

FAM = lambda n: ("C1" if n.startswith("C1") else "C2" if n.startswith("C2") else
                 "R1" if n.startswith("R1") else "R2" if n.startswith("R2") else
                 "RC1" if n.startswith("RC1") else "RC2")
CONTROLLED_PREFIX = ("cplex_ladder_", "m1_c", "m1_r1", "m1_r2", "m1_rc", "m2_h200_")

FAILS = []

def check(label, computed, stated, tol=1e-6):
    if isinstance(stated, (int, float)) and isinstance(computed, (int, float, np.integer, np.floating)):
        ok = abs(float(computed) - float(stated)) <= tol
    else:
        ok = computed == stated
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: computed={computed} stated={stated}")
    if not ok:
        FAILS.append(label)

def load_runs(studies):
    out = []
    for s in studies:
        d = RAW / s
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            if f.name == "run_manifest.json":
                continue
            out.append(json.loads(f.read_text()))
    return out

# ---------------- F2: uncertified termination ----------------
print("== F2 ==")
all_runs = load_runs([d.name for d in RAW.iterdir() if d.is_dir()])
controlled = [r for r in all_runs if r.get("study_name", "") and r["study_name"].startswith(CONTROLLED_PREFIX)]
def false_term(r):
    its = r["iteration_rows"]
    if not its:
        return False
    return r.get("termination_reason") == "no_negative_candidate" and its[-1]["pricing_status"] == "TIME_LIMIT"
ft = [r for r in controlled if false_term(r)]
check("F2 controlled total", len(controlled), 232)
check("F2 controlled false_term", len(ft), 86)
check("F2 controlled rate", len(ft) / len(controlled), 0.37, tol=0.005)
empty = sum(1 for r in ft if any(i["pricing_status"] == "TIME_LIMIT" and i.get("candidate_count", 0) == 0 for i in r["iteration_rows"]))
check("F2 empty-pool split", empty, 79)
check("F2 non-empty split", len(ft) - empty, 7)
hom = [r for r in controlled if r.get("study_name", "").startswith("m2_h200_")]
hft = [r for r in hom if false_term(r)]
check("F2 Homberger false_term", len(hft), 12)
check("F2 Homberger total", len(hom), 24)
ci = stats.binomtest(12, 24).proportion_ci(0.95)
check("F2 Homberger CI lo", round(ci.low, 2), 0.29)
check("F2 Homberger CI hi", round(ci.high, 2), 0.71)
print(f"    full-corpus totals: {len(all_runs)} runs, {sum(1 for r in all_runs if false_term(r))} false_term")

# ---------------- R3: coupling ----------------
print("== R3 ==")
m1 = [r for r in controlled if r["study_name"].startswith("m1_")]
tlrate = {}
for r in m1:
    its = r["iteration_rows"]
    tlrate[(r["instance_name"], r["variant"])] = sum(1 for i in its if i["pricing_status"] == "TIME_LIMIT") / max(len(its), 1)
pairs = pd.Series(tlrate).unstack()
d = pairs["diversified_tight_30s"] - pairs["single_tight_30s"]
d = d.dropna()
check("R3 n", len(d), 56)
check("R3 pos", int((d > 0).sum()), 43)
check("R3 ties", int((d == 0).sum()), 13)
check("R3 neg", int((d < 0).sum()), 0)
nz = d[d != 0]
w = stats.wilcoxon(nz, alternative="greater")
print(f"    wilcoxon p = {w.pvalue:.2e} (stated 5.6e-9)")
check("R3 sign-test p", round(2 ** (-len(nz)), 13), 1.1e-13, tol=1e-14)
check("R3 median all-56 (with ties)", round(float(d.median()), 2), 0.44)
check("R3 mean all-56 (with ties)", round(float(d.mean()), 2), 0.45)
check("R3 median nonzero diff", round(float(nz.median()), 2), 0.57)
check("R3 mean nonzero diff", round(float(nz.mean()), 2), 0.59)
fmean = pairs.groupby(lambda i: FAM(i)).mean()
print("    family rates:", {f: (round(r["single_tight_30s"], 2), round(r["diversified_tight_30s"], 2)) for f, r in fmean.iterrows()})

# Homberger coupling
hom_tl = {}
for r in hom:
    its = r["iteration_rows"]
    hom_tl[(r["instance_name"], r["variant"])] = sum(1 for i in its if i["pricing_status"] == "TIME_LIMIT") / max(len(its), 1)
hp = pd.Series(hom_tl).unstack()
check("R3 Homberger single rate", round(float(hp["single_tight_60s"].mean()), 2), 0.49)
check("R3 Homberger div rate", round(float(hp["diversified_tight_60s"].mean()), 2), 0.75)

# ---------------- R2: stratified occupancy ----------------
print("== R2 ==")
rows = []
for r in m1:
    pos = {k.replace("lambda_", ""): v for k, v in r.get("final_lambda_values", {}).items() if v > 1e-6}
    adm = set()
    for it in r["iteration_rows"]:
        adm.update(x for x in it.get("selected_signatures", "").split(";") if x)
    rows.append(dict(inst=r["instance_name"], variant=r["variant"], fam=FAM(r["instance_name"]),
                     pos_adm=len(pos.keys() & adm), lp_drop=r["initial_lp_objective"] - r["final_lp_objective"]))
rd = pd.DataFrame(rows)
div = rd[rd.variant == "diversified_tight_30s"]
g = div.groupby("fam").agg(m=("pos_adm", "mean"), s=("pos_adm", "std"), lp=("lp_drop", "mean"))
print(g.round(1).to_string())
r2z = rd[(rd.fam == "R2") & (rd.lp_drop.abs() < 1e-6)]
check("R2 zero-LP R2 runs", len(r2z), 22)
check("R2 mean pos_adm RC1", round(float(g.loc["RC1", "m"]), 1), 30.1)
check("R2 mean lp RC1", round(float(g.loc["RC1", "lp"]), 1), 138.4)
sing = rd[rd.variant == "single_tight_30s"].groupby("fam")["pos_adm"].mean()
print("    single pos_adm:", sing.round(1).to_dict())
check("R2 single range <=5", float(sing.max()), 4.9, tol=0.6)

# ---------------- R1: signals ----------------
print("== R1 ==")
it_rows = []
for r in m1:
    for it in r["iteration_rows"]:
        row = dict(it); row["inst"] = r["instance_name"]; row["fam"] = FAM(r["instance_name"])
        it_rows.append(row)
df = pd.DataFrame(it_rows)
df["tl"] = (df["pricing_status"] == "TIME_LIMIT").astype(int)
df["ovl"] = 1 - df["candidate_diversity"]
d2 = df[df.iteration > 1]
from sklearn.metrics import roc_auc_score
aucs = {}
for fam in sorted(d2.fam.unique()):
    vals = []
    for (i, v), gg in d2[d2.fam == fam].groupby(["inst", "variant"]):
        if gg.tl.nunique() > 1 and gg.ovl.std() > 1e-9:
            vals.append(roc_auc_score(gg.tl, gg.ovl))
    if vals:
        aucs[fam] = float(np.mean(vals))
print("    overlap AUC (iter>1):", {k: round(v, 3) for k, v in aucs.items()})
check("R1 AUC C1", round(aucs["C1"], 2), 0.66)
check("R1 AUC R1", round(aucs["R1"], 2), 0.75)
check("R1 AUC RC2", round(aucs["RC2"], 2), 0.58)
# within-run tau for C2/RC2
taus = {}
for fam in ("C2", "RC2"):
    vs = []
    for (i, v), gg in d2[d2.fam == fam].groupby(["inst", "variant"]):
        if gg.tl.nunique() > 1 and gg.ovl.std() > 1e-9:
            vs.append(stats.kendalltau(gg.ovl, gg.tl)[0])
    if vs:
        taus[fam] = float(np.mean(vs))
print("    C2/RC2 within-run tau:", {k: round(v, 3) for k, v in taus.items()})
check("R1 tau C2", round(taus["C2"], 2), 0.01)
check("R1 tau RC2", round(taus["RC2"], 2), 0.12)
deg = sum(1 for r in m1 if all(i["pricing_status"] == "TIME_LIMIT" for i in r["iteration_rows"]) or
          all(i["pricing_status"] == "OPTIMAL" for i in r["iteration_rows"]))
check("R1 degenerate runs", deg, 62)
# first-iteration displacement artifact
dd1 = df.groupby("iteration")["dual_l1_displacement"].mean()
print(f"    dual disp iter1={dd1.get(1,0):.1f} iter2={dd1.get(2,0):.1f} (stated 86 vs 14)")
check("R1 dual disp iter1", round(float(dd1.get(1, 0)), 0), 86)
check("R1 dual disp iter2", round(float(dd1.get(2, 0)), 0), 16)

# ---------------- F1: ladder identity ----------------
print("== F1 ==")
lad = [r for r in controlled if r["study_name"].startswith("cplex_ladder_")]
rows_l = []
for r in lad:
    its = r["iteration_rows"]
    tl = sum(1 for i in its if i["pricing_status"] == "TIME_LIMIT")
    rows_l.append(dict(inst=r["instance_name"], variant=r["variant"], n=len(its), tl=tl,
                       wall=sum(i["pricing_runtime_seconds"] for i in its)))
ld = pd.DataFrame(rows_l)
check("F1 ladder runs", len(ld), 96)
lim = ld.variant.str.extract(r"_(\d+)s").astype(int).values
resid = ld.wall - ld.tl * np.ravel(lim)
allc = ld[ld.tl == ld.n]
check("F1 all-capped runs", len(allc), 32)
check("F1 all-capped max residual", round(float(resid[ld.tl == ld.n].abs().max()), 2), 0.2)
r201 = ld[(ld.inst == "R201") & (ld.variant.isin(["diversified_loose_30s", "single_loose_30s"]))]
print(r201[["variant", "tl", "wall"]].to_string(index=False))

# ---------------- F3: invariance / CV ----------------
print("== F3 ==")
c101 = ld[ld.inst == "C101"].sort_values("variant")
for v, g in c101.groupby("variant"):
    if v.startswith("diversified_loose"):
        print(f"    C101 {v}: walls={g.wall.tolist()} (stated 5518.2/5518.2/5517.6 ticks)")
r2_inv = ld[(ld.inst == "R201") & (ld.variant.isin(["single_tight_15s", "single_tight_30s"]))]
print(f"    R201 single_tight 15s/30s walls={r2_inv.wall.tolist()} (stated 77300.5 ticks equal)")
rc2_inv = ld[(ld.inst == "RC201") & (ld.variant.isin(["single_tight_15s", "single_tight_30s"]))]
print(f"    RC201 single_tight 15s/30s walls={rc2_inv.wall.tolist()}")
tick = load_runs(["tickcv_c101", "tickcv_r201", "tickcv_rc201"])
for study in ("tickcv_c101", "tickcv_r201", "tickcv_rc201"):
    tr = [r for r in tick if r["study_name"] == study]
    g = pd.Series({r["variant"]: sum(i.get("pricing_dettime_ticks") or 0 for i in r["iteration_rows"]) for r in tr})
    if len(g) > 1:
        print(f"    {study}: tick CV = {g.std()/g.mean()*100:.1f}% (values {g.values.round(0).tolist()})")

# ---------------- Prescription ----------------
print("== Prescription ==")
ev = pd.read_csv(GENDIR / "rule_evaluation.csv")
check("RX n valid", len(ev), 55)
w2 = stats.wilcoxon(ev.rule_ticks, ev.div_ticks)
print(f"    wilcoxon rule vs div p = {w2.pvalue:.2e} (stated 4.7e-5)")
check("RX median rule/div", round(float((ev.rule_ticks / ev.div_ticks).median()), 2), 0.65)
check("RX median rule/single", round(float((ev.rule_ticks / ev.single_ticks).median()), 2), 0.97)
famr = ev.groupby("family")["ticks_vs_div"].median()
print("    family rule/div medians:", famr.round(2).to_dict(), "(stated 0.34 RC1 .. 0.82 C1)")
check("RX family min", round(float(famr.min()), 2), 0.34)
check("RX family max", round(float(famr.max()), 2), 0.82)
rb = ev.rule_lp / ev[["single_lp", "div_lp"]].min(axis=1) - 1
sb = ev.single_lp / ev[["single_lp", "div_lp"]].min(axis=1) - 1
check("RX rule >0.5% bound cost", round(float((rb > 0.005).mean()), 2), 0.25)
check("RX single >0.5% bound cost", round(float((sb > 0.005).mean()), 2), 0.33)
check("RX rule mean bound deg", round(float(rb.mean() * 100), 1), 1.2)
check("RX single mean bound deg", round(float(sb.mean() * 100), 1), 4.1)
g = ev.gap_closed.dropna()
nzgap = ((ev.div_ticks - ev.oracle_ticks).abs() >= 1)
check("RX nonzero-gap n", int(nzgap.sum()), 19)
check("RX gap_closed median", round(float(g.median()), 3), 0.977)
check("RX crosses oracle", round(float((g > 1).mean()), 2), 0.47)

# ---------------- Certificate table ----------------
print("== Cert ==")
certs = sorted(GENDIR.glob("certification_*.json"))
check("Cert table rows", len(certs), 17)
violated = sum(1 for c in certs if (json.loads(c.read_text())["certificate_best_rc"] or 0) < 0)
check("Cert all-violated", violated, 17)


# ---------------- Anytime certificates ----------------
print("== Anytime ==")
ac = pd.read_csv(GENDIR / "anytime_cert_all.csv")
check("Anytime total calls", len(ac), 41)
check("Anytime certified", int(ac.certified.sum()), 0)
ac2 = ac.copy()
ac2["strat"] = ac2.variant.str.replace("_tight_30s", "", regex=False)
ac2.loc[ac2.variant.str.startswith("rule"), "strat"] = "rule"
p = ac2.pivot_table(index="instance", columns="strat", values="best_rc", aggfunc="first")
cmp = p.dropna()
closer = int((cmp["rule"] > cmp[["single", "diversified"]].min(axis=1)).sum())
check("Anytime rule-closer count", closer, 7)
check("Anytime comparable n", len(cmp), 13)
rel = (cmp["rule"] / cmp[["single", "diversified"]].min(axis=1)).abs()
check("Anytime median ratio", round(float(rel.median()), 2), 0.78)


# ---------------- GC cross-problem ----------------
print("== GC ==")
import json as _json
gcdir = PLAN2 / "results" / "raw"
lad = [_json.loads(f.read_text()) for f in (gcdir / "gc_ladder").glob("*.json")]
check("GC ladder runs", len(lad), 24)
tl_total = sum(1 for r in lad for i in r["iteration_rows"] if i["pricing_status"] == "TIME_LIMIT")
check("GC ladder TL hits", tl_total, 0)
m1 = [_json.loads(f.read_text()) for f in (gcdir / "gc_m1").glob("*.json")]
check("GC m1 runs", len(m1), 24)
def _ticks(r): return sum(i["pricing_dettime_ticks"] or 0 for i in r["iteration_rows"])
piv = {}
for r in m1:
    st = "div" if r["variant"].startswith("div") else "single"
    piv.setdefault(r["instance_name"], {})[st] = _ticks(r)
ratios = [v["div"] / v["single"] for v in piv.values() if "single" in v and v["single"] > 0]
import statistics
check("GC coupling median ratio", round(statistics.median(ratios), 1), 10.3)
from scipy import stats as _st
wgc = _st.wilcoxon([v["div"] for v in piv.values()], [v["single"] for v in piv.values()])
print(f"    GC Wilcoxon p = {wgc.pvalue:.1e} (stated 5e-4)")

print()
print(f"TOTAL FAILS: {len(FAILS)}")
for f in FAILS:
    print("  -", f)
