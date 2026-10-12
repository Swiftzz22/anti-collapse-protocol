"""Generates every data figure in the paper from the real experiment logs (ACL-styled versions).

    python make_figures_acl.py

Output files (in ../figures/) -- the names deliberately differ from the earlier figure files:
    main_selfconsumption_overview   (Fig. hero)          main_recursive_dynamics   (dashboard)
    main_consensus_mechanisms       (mechanisms)         main_openweb_health       (Regime II)
    app_screening_funnel            app_screening_per_round
    app_gate_t1_entropy             app_gate_t2_tail     app_gate_t3_isotropy      app_gate_t4_routes
    app_accuracy_trajectories       app_openweb_per_round
    app_route_entropy_probes        app_drift_and_loss
"""
import json, os
from collections import Counter
import numpy as np
from scipy.stats import pearsonr, gaussian_kde
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from style_acl import *

B = load_B(); A = load_A()
NV = json.load(open(os.path.join(HERE, "naive.json")))
RHO = dict((k, r) for k, r in json.load(open(os.path.join(HERE, "rho.json"))))
REP = json.load(open(os.path.join(HERE, "rep_sizecheck.json")))
models = ["gemma2-9b", "llama3.1-8b", "mistral-7b", "qwen2.5-7b"]
tasks = ["gsm8k", "arc_challenge", "commonsense_qa", "boolq"]
SET = [(m, t) for m in models for t in tasks]
R = np.arange(9)
G = np.arange(1, 9)
BBN = ["Gemma-2-9B", "Llama-3-8B", "Mistral-7B-v0.3", "Qwen2.5-7B"]
BBD = {"Gemma-2-9B": "openweb_gemma", "Llama-3-8B": "openweb_llama",
       "Mistral-7B-v0.3": "openweb_mistral", "Qwen2.5-7B": "openweb_qwen"}


def model_task_legend(ax, loc="lower left", ncol=3, **kw):
    h = [Line2D([], [], color=C_MODEL[m], lw=1.3, label=S_MODEL[m]) for m in models]
    h += [Line2D([], [], color="#555", lw=0, marker=M_TASK[t], ms=3.4, label=N_TASK[t]) for t in tasks]
    return h


# ====================================================================== 1. HERO / OVERVIEW
fig = plt.figure(figsize=(FULL, 2.15))
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.18, 1, 0.95], wspace=0.38, hspace=0.18)

ax = fig.add_subplot(gs[:, 0])
allD = []
for m in models:
    for t in tasks:
        a = np.array(B[f"{m}|{t}"]["acc"]) * 100
        d = a - a[0]; allD.append(d)
        ax.plot(R, d, color=C_MODEL[m], lw=0.65, alpha=0.6, marker=M_TASK[t], ms=2.1, mew=0, zorder=2)
allD = np.array(allD)
ax.axhspan(-2.0, 2.0, color="#F0F0F0", zorder=0, lw=0)
ax.axhline(0, color="#555", lw=0.5, ls=(0, (2, 2)), zorder=1)
mean_line(ax, R, allD.mean(0), "black", lw=1.8, ms=3.4)
ax.set_xlabel("self-consumption generation"); ax.set_ylabel(r"$\Delta$ accuracy vs. base (pp)")
ax.set_xticks(R); ax.set_ylim(-6.2, 3.1); ax.set_xlim(-0.3, 8.3)
ax.text(0.2, 2.25, r"typical 95% CI ($\pm$2 pp)", fontsize=5.9, color="#777")
h = model_task_legend(ax) + [Line2D([], [], color="k", lw=1.8, label="mean")]
ax.legend(handles=h, ncol=3, loc="lower left", fontsize=5.7, handlelength=1.1, columnspacing=0.7,
          handletextpad=0.3, borderaxespad=0.2)
ptitle(ax, "a", "accuracy over 8 label-free generations")

ax = fig.add_subplot(gs[:, 1])
gp, ap = [], []
ax.fill_between([46, 100], [46, 100], [100, 100], color="#EEF3F9", zorder=0, lw=0)
for t in tasks:
    for m in models:
        v = B[f"{m}|{t}"]
        ax.scatter(np.array(v["gen_prec"]) * 100, np.array(v["acc_prec"]) * 100, s=9, color=C_TASK[t],
                   marker=M_TASK[t], lw=0.25, edgecolors="white", alpha=0.95, zorder=3)
        gp += v["gen_prec"]; ap += v["acc_prec"]
gp, ap = np.array(gp) * 100, np.array(ap) * 100
ax.plot([46, 100], [46, 100], color="#444", lw=0.6, ls=(0, (3, 2)), zorder=2)
ax.text(63.6, 66.4, "$y{=}x$", fontsize=6.2, rotation=45, rotation_mode="anchor", transform_rotates_text=True,
        color="#555", ha="center", va="bottom")
ax.text(60, 97.5, "admitted purer", fontsize=5.9, color=C_ACC, style="italic", va="top")
ax.set_xlim(48, 100); ax.set_ylim(48, 100)
ax.set_xlabel("label precision of generated data (%)"); ax.set_ylabel("label precision after ACP (%)")
r, p = pearsonr(gp, ap - gp)
ax.text(51, 52, f"128/128 rounds above $y{{=}}x$\nlift vs. generator: $r={r:.2f}$", fontsize=6.0, va="bottom")
h = [Line2D([], [], color=C_TASK[t], lw=0, marker=M_TASK[t], ms=3.6, mec="white", mew=0.3, label=N_TASK[t])
     for t in tasks]
ax.legend(handles=h, loc="center left", fontsize=5.9, handletextpad=0.2, ncol=1, bbox_to_anchor=(0.0, 0.58))
ax.grid(True, axis="both")
ptitle(ax, "b", "admitted vs. generated precision")

nacc = np.array(NV["acc"]) * 100
aacc = np.array(B["mistral-7b|gsm8k"]["acc"]) * 100
ax1 = fig.add_subplot(gs[0, 2])
mean_line(ax1, range(5), nacc, C_NAIVE, marker="o", lw=1.3, ms=3.0, label="naive loop")
mean_line(ax1, range(9), aacc, C_ACP, marker="s", lw=1.3, ms=3.0, label="ACP")
ax1.set_ylim(53, 60.5); ax1.set_xticks(R); ax1.set_xticklabels([]); ax1.set_xlim(-0.4, 8.4)
ax1.set_ylabel("GSM8K acc. (%)", fontsize=6.8)
ax1.legend(loc="lower right", fontsize=5.9, ncol=2, columnspacing=0.8)
ptitle(ax1, "c", "Mistral/GSM8K vs. naive")
ax2 = fig.add_subplot(gs[1, 2])
st = B["mistral-7b|gsm8k"]
wrong_a, fa = 0, []
for n, pr, pool in zip(st["final"], st["acc_prec"], st["pool"]):
    wrong_a += n * (1 - pr); fa.append(100 * wrong_a / pool)
wrong_n, fn = 0, []
for s in NV["stats"]:
    wrong_n += s["num_accepted"] * (1 - s["accepted_label_precision"]); fn.append(100 * wrong_n / s["pool_size"])
ax2.bar(np.arange(1, 5) - 0.19, fn, width=0.38, color=C_NAIVE, lw=0)
ax2.bar(np.arange(1, 9) + (np.arange(1, 9) <= 4) * 0.19, fa, width=0.38, color=C_ACP, lw=0)
ax2.set_xticks(R); ax2.set_xlim(-0.4, 8.4); ax2.set_ylim(0, 58)
ax2.set_ylabel("wrong labels in\ntraining pool (%)", fontsize=6.8); ax2.set_xlabel("generation")
ax2.text(2.5, 52, f"{fn[-1]:.0f}%", color=C_NAIVE, fontsize=6.6, fontweight="bold", ha="center")
ax2.text(6.5, 14, f"{fa[3]:.1f}%", color=C_ACP, fontsize=6.6, fontweight="bold", ha="center")
save(fig, "main_selfconsumption_overview")

# ====================================================================== 2. MECHANISMS
fig, axs = plt.subplots(1, 3, figsize=(FULL, 1.78), gridspec_kw={"wspace": 0.36, "width_ratios": [1.12, 1.15, 1]})
ax = axs[0]
w = 0.19
for i, m in enumerate(models):
    vals = [RHO[f"{m}|{t}"] for t in tasks]
    ax.bar(np.arange(4) + (i - 1.5) * w, vals, width=w * 0.92, color=C_MODEL[m], label=S_MODEL[m], lw=0)
ax.axhline(0, color="#444", lw=0.55)
ax.set_xticks(range(4)); ax.set_xticklabels([N_TASK[t] for t in tasks])
ax.tick_params(axis="x", length=0)
ax.set_ylabel(r"effective common-mode $\hat\rho$"); ax.set_ylim(-0.33, 1.2)
ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
ax.legend(fontsize=5.8, loc="upper left", ncol=2, handlelength=0.9, columnspacing=0.8, handletextpad=0.3)
ptitle(ax, "a", "route-error correlation by task")

ax = axs[1]
Gp = np.array([B[f"{m}|{t}"]["gen_prec"] for m, t in SET]) * 100
Ap = np.array([B[f"{m}|{t}"]["acc_prec"] for m, t in SET]) * 100
G0 = Gp - Gp[:, :1]; A0 = Ap - Ap[:, :1]
thin(ax, G, G0, C_GEN, alpha=0.35)
thin(ax, G, A0, C_ACC, alpha=0.22)
ax.fill_between(G, G0.mean(0), A0.mean(0), color=C_ACC, alpha=0.10, lw=0)
mean_line(ax, G, G0.mean(0), "#4A4A4A", marker="o", label="generated (student's own)")
mean_line(ax, G, A0.mean(0), C_ACC, marker="s", label="admitted by ACP")
ax.axhline(0, color="#555", lw=0.5, ls=(0, (2, 2)))
ax.set_xticks(G); ax.set_xlabel("round"); ax.set_ylabel("label precision vs. round 1 (pp)")
ax.set_ylim(-10.5, 6.5); ax.set_yticks([-10, -5, 0, 5]); ax.legend(fontsize=5.9, loc="lower left")
ptitle(ax, "b", "the generator erodes; admission holds")

ax = axs[2]
dr, dd = [], []
for m in models:
    for t in tasks:
        v = B[f"{m}|{t}"]; c = [(b[0] + b[1]) / 2 for b in v["band"]]
        x = 100 * (c[-1] / c[0] - 1); y = 100 * v["delta"]; dr.append(x); dd.append(y)
        ax.scatter(x, y, color=C_MODEL[m], marker=M_TASK[t], s=17, lw=0.35, edgecolors="white", zorder=3)
r, p = pearsonr(dr, dd)
ax.axhline(0, color="#555", lw=0.5, ls=(0, (2, 2)))
ax.set_xlabel(r"drift of $H_{\pi_0}$ band centre (%)"); ax.set_ylabel(r"$\Delta$acc G8$-$G0 (pp)")
ax.set_ylim(-4.3, 2.75); ax.set_xlim(-8, 92)
ax.text(0.97, 0.05, f"$r={r:.2f}$, $p={p:.2f}$", fontsize=6.1, transform=ax.transAxes, ha="right")
h = [Line2D([], [], color=C_MODEL[m], lw=0, marker="o", ms=3.4, label=S_MODEL[m]) for m in models]
ax.legend(handles=h, fontsize=5.6, loc="upper left", ncol=4, columnspacing=0.45, handletextpad=0.05,
          borderaxespad=0.15)
ax.grid(True, axis="both")
ptitle(ax, "c", r"drift $\neq$ collapse")
print("drift-delta corr", r, p, "drift range", min(dr), max(dr))
save(fig, "main_consensus_mechanisms")

# ====================================================================== 3. OPEN WEB (Regime II)
bbs = list(A.keys())
fig, axs = plt.subplots(1, 4, figsize=(FULL, 1.72), gridspec_kw={"wspace": 0.48, "width_ratios": [1, 1, 1.08, 1.3]})


def paired(ax, key, ylabel, scale=1.0):
    for i, b in enumerate(bbs):
        g = np.array(A[b]["ml"]["generated"][key]) * scale; a = np.array(A[b]["ml"]["accepted"][key]) * scale
        ax.plot([i - 0.16, i + 0.16], [g.mean(), a.mean()], color="#C8C8C8", lw=0.8, zorder=1)
        ax.errorbar(i - 0.16, g.mean(), yerr=g.std(), fmt="o", color=C_GEN, ms=3.4, lw=0.7, capsize=1.6,
                    mec="white", mew=0.4, zorder=3)
        ax.errorbar(i + 0.16, a.mean(), yerr=a.std(), fmt="s", color=C_BB[b], ms=3.4, lw=0.7, capsize=1.6,
                    mec="white", mew=0.4, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Gemma", "Llama", "Mistral", "Qwen"], fontsize=6.0, rotation=30,
                                             ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0); ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel(ylabel)


paired(axs[0], "tail_mass", "Zipf-tail mass (%)")
ptitle(axs[0], "a", r"tail support $\uparrow$")
paired(axs[1], "entropy_std", r"s.d. of $H_{\pi_0}$ (bits)")
axs[1].set_ylim(0, 1.05)
ptitle(axs[1], "b", "entropy dispersion")
h = [Line2D([], [], color=C_GEN, marker="o", lw=0, ms=3.4, label="generated"),
     Line2D([], [], color="#333", marker="s", lw=0, ms=3.4, label="admitted")]
axs[1].legend(handles=h, fontsize=5.9, loc="upper right", handletextpad=0.1)
ax = axs[2]
rr = np.array(REP); x = rr[:, 0]
ax.fill_between(x, rr[:, 2], rr[:, 1], color="#44AA99", alpha=0.12, lw=0)
mean_line(ax, x, rr[:, 3], C_GEN, marker="o", lw=1.1, ms=2.8, label="generated")
mean_line(ax, x, rr[:, 1], "#44AA99", marker="s", lw=1.0, ms=2.6, ls=(0, (3, 1.5)), label="admitted (raw)")
mean_line(ax, x, rr[:, 2], "#227755", marker="^", lw=1.1, ms=2.8, label="admitted, size-matched")
ax.text(4.5, 0.418, "size artifact", fontsize=5.6, color="#227755", style="italic", ha="center")
ax.set_xlabel("round"); ax.set_ylabel("bigram repetition"); ax.set_xticks(range(1, 9))
ax.set_ylim(0.30, 0.53)
ax.legend(fontsize=5.4, loc="lower right", handlelength=1.6, borderaxespad=0.1)
ptitle(ax, "c", "repetition (Qwen), audited")
ax = axs[3]
routes = ["direct", "paraphrase", "expand", "contrast", "socratic"]
M = np.zeros((2, 5))
for b in bbs:
    tr = A[b]["teacher_route"]
    for i, t in enumerate(["teacher_primary", "teacher_mistral"]):
        for j, r_ in enumerate(routes):
            M[i, j] += tr.get(f"{t}|{r_}", 0) / 320 / 4
cm = LinearSegmentedColormap.from_list("acpb", ["#F4F8FC", "#9CC0E0", "#1F4E79"])
ax.imshow(M * 100, cmap=cm, vmin=12, vmax=85, aspect="auto")
for i in range(2):
    for j in range(5):
        ax.text(j, i, f"{100*M[i,j]:.0f}", ha="center", va="center", fontsize=6.8,
                color="white" if M[i, j] > 0.52 else INK)
for k in np.arange(-0.5, 5, 1): ax.axvline(k, color="white", lw=1.2)
ax.axhline(0.5, color="white", lw=1.2)
ax.set_xticks(range(5)); ax.set_xticklabels(["dir.", "para.", "exp.", "contr.", "Socr."], fontsize=5.9)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Qwen-I", "Mistral-I"], fontsize=6.1)
ax.tick_params(length=0); ax.grid(False)
for s in ax.spines.values(): s.set_visible(False)
ptitle(ax, "d", r"admit rate (%), teacher $\times$ route")
save(fig, "main_openweb_health")

# ====================================================================== 4. FUNNEL (both regimes, mean)
fig, ax = plt.subplots(figsize=(COL * 1.25, 1.12))
keys = ["consensus", "T1 band", "T2 novelty", "dedup cap", "T3 / T4 caps", "admitted"]
cols = [C_FUN["consensus"], C_FUN["T1"], C_FUN["T2"], C_FUN["dedup"], C_FUN["T34"], C_FUN["admitted"]]
fI = np.zeros(6)
for v in B.values():
    fI += np.array([2000 - np.mean(v["after_corr"]), np.mean(v["rej_t1"]), np.mean(v["rej_t2"]), np.mean(v["rej_dup"]),
                    np.mean(v["rej_t3"]) + np.mean(v["rej_crit"]) + np.mean(v["rej_route"]), np.mean(v["final"])]) / len(B)
fI = fI / 2000 * 100
fII = np.zeros(6)
for b in bbs:
    dg = A[b]["diag"]
    fII += np.array([np.mean([x["num_generated"] - x["input_candidates"] for x in dg]), np.mean([x["reject_t1"] for x in dg]),
                     np.mean([x["reject_t2"] for x in dg]), np.mean([x["reject_commit_dup"] for x in dg]),
                     0, np.mean([x["num_accepted"] for x in dg])]) / np.mean([x["num_generated"] for x in dg]) / 4 * 100
for row, f in enumerate([fII, fI]):
    left = 0
    for k, (val, c) in enumerate(zip(f, cols)):
        ax.barh(row, val, left=left, color=c, height=0.58, label=keys[k] if row == 0 else None, lw=0.6,
                edgecolor="white")
        if val > 4:
            ax.text(left + val / 2, row, f"{val:.0f}", ha="center", va="center", fontsize=6.4,
                    color="white" if k in (1, 5) else INK)
        left += val
ax.set_yticks([0, 1]); ax.set_yticklabels(["II: open web", "I: self-consuming"], fontsize=6.8)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 100); ax.set_xlabel("share of generated candidates (%)")
ax.grid(True, axis="x"); ax.grid(False, axis="y")
ax.spines["left"].set_visible(False)
ax.legend(ncol=6, fontsize=5.7, loc="lower center", bbox_to_anchor=(0.45, 1.0), columnspacing=0.8,
          handlelength=0.9, handletextpad=0.3)
print("funnel I", fI.round(1), "II", fII.round(1))
save(fig, "app_screening_funnel")


# ====================================================================== 5. RECURSIVE-DYNAMICS DASHBOARD (Regime I)
gp = np.array([B[f"{m}|{t}"]["gen_prec"] for m, t in SET]) * 100
ap = np.array([B[f"{m}|{t}"]["acc_prec"] for m, t in SET]) * 100
acc = np.array([B[f"{m}|{t}"]["acc"] for m, t in SET]) * 100
dacc = acc - acc[:, :1]
band = np.array([B[f"{m}|{t}"]["band"] for m, t in SET])          # (16, 8, 2)
rH = np.array([B[f"{m}|{t}"]["route_H"] for m, t in SET])
cf = np.array([B[f"{m}|{t}"]["cf"] for m, t in SET])
pool = np.array([B[f"{m}|{t}"]["pool"] for m, t in SET])
loss = np.array([B[f"{m}|{t}"]["loss"] for m, t in SET])
fin = np.array([B[f"{m}|{t}"]["final"] for m, t in SET])
after = np.array([B[f"{m}|{t}"]["after_corr"] for m, t in SET])
r1 = np.array([B[f"{m}|{t}"]["rej_t1"] for m, t in SET])
r2 = np.array([B[f"{m}|{t}"]["rej_t2"] for m, t in SET])
rd = np.array([B[f"{m}|{t}"]["rej_dup"] for m, t in SET])
rcon = 2000 - after


def stage(ax, name, sub):
    """Stage tag (small caps in the stage colour, no box) + panel subtitle."""
    ax.text(0.0, 1.20, name.upper(), transform=ax.transAxes, fontsize=5.9, color=ST[name], fontweight="bold",
            va="bottom", ha="left")
    ax.plot([0.0, 1.0], [1.185, 1.185], transform=ax.transAxes, color=ST[name], lw=0.9, clip_on=False)
    ax.text(0.0, 1.03, sub, transform=ax.transAxes, fontsize=6.6, va="bottom", ha="left")


fig = plt.figure(figsize=(FULL, 3.55))
gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.46, hspace=0.80)

ax = fig.add_subplot(gs[0, 0]); stage(ax, "Synthesize", "student's own label precision (%)")
thin(ax, G, gp, "#555555", alpha=0.25)
mean_line(ax, G, gp.mean(0), "#3A3A3A")
ax.set_xticks(G); ax.set_ylim(50, 100)
_tr = np.array([np.polyfit(np.arange(8), r, 1)[0] * 7 for r in gp])
note(ax, 0.97, 0.04, f"linear trend R1→R8: {_tr.mean():+.1f} pp\n(negative in {(_tr<0).sum()}/16)", ha="right",
     va="bottom", fs=5.6, color="#444")

ax = fig.add_subplot(gs[0, 1]); stage(ax, "Screen", "consensus: admitted precision (%)")
thin(ax, G, ap, C_ACC, alpha=0.18)
ax.fill_between(G, gp.mean(0), ap.mean(0), color=C_ACC, alpha=0.12, lw=0)
ax.plot(G, gp.mean(0), color="#3A3A3A", lw=1.0, ls=(0, (3, 1.5)), label="generated (mean)")
mean_line(ax, G, ap.mean(0), C_ACC, marker="s", label="admitted (mean)")
ax.set_xticks(G); ax.set_ylim(50, 100)
note(ax, 0.97, 0.04, "adm. > gen. in 128/128\nnoise 20.0% → 11.0%", ha="right", va="bottom", fs=5.6, color=C_ACC)

ax = fig.add_subplot(gs[0, 2]); stage(ax, "Screen", r"T1 band on $H_{\pi_0}$ (bits/token)")
for m in models:
    idx = [i for i, (mm, _) in enumerate(SET) if mm == m]
    lo = band[idx, :, 0].mean(0); hi = band[idx, :, 1].mean(0)
    ax.fill_between(G, lo, hi, color=C_MODEL[m], alpha=0.10, lw=0)
    mean_line(ax, G, (lo + hi) / 2, C_MODEL[m], lw=1.2, ms=2.6)
ax.set_xticks(G)
h = [Line2D([], [], color=C_MODEL[m], lw=1.3, label=S_MODEL[m]) for m in models]
ax.legend(handles=h, fontsize=5.4, ncol=2, loc="lower right", handlelength=1.0, columnspacing=0.6)
ax.set_ylim(0, 1.45)

ax = fig.add_subplot(gs[0, 3]); stage(ax, "Screen", "where candidates go (% of 2,000)")
shares = np.stack([rcon, r1, r2 + rd, fin]).mean(1) / 20.0   # (4, 8)
cols = [C_FUN["consensus"], C_FUN["T1"], C_FUN["T2"], C_FUN["admitted"]]
labs = ["consensus", "T1 band", "T2 + dedup", "admitted"]
ax.stackplot(G, shares, colors=cols, labels=labs, lw=0.5, edgecolor="white")
ax.set_xticks(G); ax.set_ylim(0, 100); ax.set_xlim(1, 8)
ax.grid(False)
ax.legend(fontsize=5.4, loc="upper center", ncol=4, bbox_to_anchor=(0.45, -0.12), handlelength=0.8,
          columnspacing=0.5, handletextpad=0.25)

ax = fig.add_subplot(gs[1, 0]); stage(ax, "Screen", r"T4: admitted route entropy (bits)")
thin(ax, G, rH, ST["Measure"], alpha=0.25)
mean_line(ax, G, rH.mean(0), ST["Measure"])
ax.axhline(np.log2(5), color="#333", lw=0.6, ls=(0, (1, 1.5)))
ax.text(1.0, np.log2(5) + 0.002, r"max $=\log_2 5$", fontsize=5.6, va="bottom")
ax.set_ylim(2.24, 2.338); ax.set_xticks(G)
note(ax, 0.97, 0.05, "tripwire at 1 bit: never fired", ha="right", va="bottom", fs=5.6, color=ST["Adapt"])
ax.set_xlabel("generation")

ax = fig.add_subplot(gs[1, 1]); stage(ax, "Train", "cumulative pool (k), LoRA loss")
thin(ax, G, pool / 1000, ST["Train"], alpha=0.22)
mean_line(ax, G, (pool / 1000).mean(0), ST["Train"])
ax.set_xticks(G); ax.set_xlabel("generation"); ax.set_ylim(0, 12); ax.set_yticks([0, 4, 8, 12])
ax.set_ylabel("pool (k)", fontsize=6.2, color=ST["Train"])
ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
ax2.plot(G, np.nanmean(loss, 0), color="#333", lw=1.0, ls=(0, (3, 1.5)), marker="s", ms=2.4, mec="white", mew=0.5)
ax2.tick_params(labelsize=6.0); ax2.set_ylim(0.18, 0.37); ax2.set_yticks([0.2, 0.25, 0.3, 0.35])
ax.legend(handles=[Line2D([], [], color=ST["Train"], lw=1.5, label="pool size"),
                   Line2D([], [], color="#333", lw=1, ls=(0, (3, 1.5)), label="train loss")], fontsize=5.4,
          loc="upper center")

ax = fig.add_subplot(gs[1, 2]); stage(ax, "Measure", "counterfactual-probe catch rate")
thin(ax, G, cf, ST["Measure"], alpha=0.22)
mean_line(ax, G, cf.mean(0), ST["Measure"])
ax.set_ylim(0.3, 1.0); ax.set_xticks(G); ax.set_xlabel("generation")
note(ax, 0.98, 0.04, f"mean {cf.mean():.2f} (range {cf.mean(1).min():.2f}–{cf.mean(1).max():.2f})", ha="right",
     va="bottom", fs=5.3)

ax = fig.add_subplot(gs[1, 3]); stage(ax, "Measure", r"outcome: $\Delta$ accuracy vs. G0 (pp)")
ax.axhspan(-2, 2, color="#F0F0F0", lw=0, zorder=0)
for row, (m, t) in zip(dacc[:, 1:], SET):
    ax.plot(G, row, color=C_MODEL[m], lw=0.5, alpha=0.5)
mean_line(ax, G, dacc[:, 1:].mean(0), "black")
ax.axhline(0, color="#555", lw=0.5, ls=(0, (2, 2)))
ax.set_ylim(-6.6, 2.8); ax.set_xticks(G); ax.set_xlabel("generation")
note(ax, 0.98, 0.04, "mean G8: −0.72 pp\n14/16 n.s.", ha="right", va="bottom", fs=5.3)
save(fig, "main_recursive_dynamics")

# ====================================================================== 6. T1 entropy banding
fig = plt.figure(figsize=(FULL, 1.85))
gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.30, width_ratios=[1, 1, 1, 1, 0.10, 1.55], top=0.80)
for i, bb in enumerate(BBN):
    ax = fig.add_subplot(gs[0, i])
    rows = [json.loads(l) for l in open(os.path.join(RAW, BBD[bb], "accepted.jsonl"))]
    e = np.array([float(r["entropy"]) for r in rows if str(r.get("is_counterfactual", "False")) == "False"])
    ml = A[bb]["ml"]
    mu_g, sd_g = np.mean(ml["generated"]["entropy_mean"]), np.mean(ml["generated"]["entropy_std"])
    xs = np.linspace(mu_g - 2.4 * sd_g, mu_g + 2.4 * sd_g, 300)
    ax.axvspan(mu_g - sd_g, mu_g + sd_g, color="#E4E4E4", lw=0)
    ax.axvline(mu_g, color="#888", lw=0.6, ls=(0, (2, 2)))
    k = gaussian_kde(e)
    ax.fill_between(xs, k(xs), color=C_BB[bb], alpha=0.30, lw=0)
    ax.plot(xs, k(xs), color=C_BB[bb], lw=1.1)
    sda = np.mean(ml["accepted"]["entropy_std"])
    ax.set_title(S_BB[bb], fontsize=6.8, loc="left")
    note(ax, 0.97, 0.97, f"s.d. {sd_g:.2f}→{sda:.2f}", ha="right", va="top", fs=5.6)
    ax.set_ylim(0, k(xs).max() * 1.35)
    ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"$H_{\pi_0}$ (bits/token)", fontsize=6.4); ax.set_xlim(xs[0], xs[-1])
    if i == 0:
        ax.set_ylabel("density")
        fig.legend(handles=[Rectangle((0, 0), 1, 1, fc="#E4E4E4"), Line2D([], [], color="#333", lw=1.1)],
                   labels=["generated pool: logged mean ± s.d.", "admitted samples (KDE of all logged)"],
                   fontsize=5.8, loc="upper left", bbox_to_anchor=(0.06, -0.01), ncol=2, handlelength=1.2)
fig.text(0.085, 0.905, r"$\mathbf{(a)}$ Regime II: admitted text concentrates in the learnable band", fontsize=7.6,
         va="bottom")
ax = fig.add_subplot(gs[0, 5])
for t in tasks:
    idx = [i for i, (_, tt) in enumerate(SET) if tt == t]
    lo = band[idx, :, 0].mean(0); hi = band[idx, :, 1].mean(0)
    ax.fill_between(G, lo, hi, color=C_TASK[t], alpha=0.13, lw=0)
    ax.plot(G, lo, color=C_TASK[t], lw=0.5, alpha=0.8); ax.plot(G, hi, color=C_TASK[t], lw=0.5, alpha=0.8)
    mean_line(ax, G, (lo + hi) / 2, C_TASK[t], marker=M_TASK[t], lw=1.3, ms=3.0, label=N_TASK[t])
ax.set_xticks(G); ax.set_xlabel("generation"); ax.set_ylabel(r"$H_{\pi_0}$ band (bits/token)", fontsize=6.6)
ax.set_ylim(0, 2.65)
ax.legend(fontsize=5.8, ncol=2, loc="upper left", handlelength=1.6, columnspacing=1.0, handletextpad=0.3)
ptitle(ax, "b", r"Regime I: band [Q$_{10}$, Q$_{90}$] drifts upward")
save(fig, "app_gate_t1_entropy")

# ====================================================================== 7. T2 tail support
fig, axs = plt.subplots(1, 4, figsize=(FULL, 1.62), sharey=True)
fig.subplots_adjust(wspace=0.10)
for ax, bb in zip(axs, BBN):
    ml = A[bb]["ml"]
    tg = np.array(ml["generated"]["tail_mass"]); ta = np.array(ml["accepted"]["tail_mass"])
    ax.fill_between(G, tg, ta, where=ta >= tg, color=C_BB[bb], alpha=0.14, lw=0, interpolate=True)
    mean_line(ax, G, tg, C_GEN, marker="o", lw=1.0, ms=2.8, label="generated")
    mean_line(ax, G, ta, C_BB[bb], marker="s", lw=1.4, ms=2.9, label="admitted")
    ax.set_title(S_BB[bb], fontsize=6.9, loc="left")
    note(ax, 0.97, 0.97, f"×{ta.mean()/tg.mean():.2f};  {int((ta>tg).sum())}/8 rounds", ha="right", va="top", fs=5.8,
         color=C_BB[bb])
    ax.set_xticks(G); ax.set_xlabel("round")
axs[0].set_ylabel("Zipf-tail mass (%)"); axs[0].set_ylim(0.58, 1.55)
fig.legend(handles=[Line2D([], [], color=C_GEN, marker="o", ms=2.8, lw=1.0, label="generated"),
                    Line2D([], [], color="#333", marker="s", ms=2.8, lw=1.4, label="admitted (backbone colour)")],
           fontsize=6.0, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.02))
save(fig, "app_gate_t2_tail")

# ====================================================================== 8. T3 coverage isotropy
cmap = LinearSegmentedColormap.from_list("iso", ["#2F6DA8", "#F7F7F7", "#B0403A"])
fig = plt.figure(figsize=(FULL, 2.0))
gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.32, width_ratios=[1, 1, 1, 1, 0.45, 1.35])
KL = {}
for i, bb in enumerate(BBN):
    rows = [json.loads(l) for l in open(os.path.join(RAW, BBD[bb], "accepted.jsonl"))]
    rows = [r for r in rows if str(r.get("is_counterfactual", "False")) == "False"]
    C = int(max(int(r["cluster_id"]) for r in rows)) + 1
    M = np.zeros((C, 8))
    for r in rows:
        M[int(r["cluster_id"]), int(r["round_id"]) - 1] += 1
    P = M / M.sum(); pd_ = P.sum(1, keepdims=True); pw = P.sum(0, keepdims=True)
    ratio = P / (pd_ @ pw + 1e-12)
    eps = 1e-3
    kl = []
    for w in range(8):
        q = (M[:, w] + eps) / (M[:, w].sum() + eps * C); p = (pd_.ravel() + eps) / (1 + eps * C)
        kl.append(float(np.sum(q * np.log(q / p))))
    KL[bb] = kl
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(ratio, cmap=cmap, norm=TwoSlopeNorm(1.0, 0.0, 2.0), aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels([f"{w+1}" for w in range(8)], fontsize=5.6)
    ax.set_yticks(range(C)); ax.set_yticklabels([f"c{d}" for d in range(C)], fontsize=5.4)
    ax.tick_params(length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True); ax.set_yticks(np.arange(-0.5, C, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.5); ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_title(S_BB[bb], fontsize=6.8, loc="left")
    ax.set_xlabel("round", fontsize=6.2)
    last_hm = ax
pos = last_hm.get_position()
cax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.007, pos.height]); cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=5.4, length=1.5); cb.outline.set_visible(False)
fig.text(0.085, 1.0, r"$\mathbf{(a)}$ admitted coverage ratio $r(d,w)=\hat p(d,w)/\hat p(d)\hat p(w)$ per seed cluster "
         r"$d$ and round $w$ (1.0 = isotropic)", fontsize=7.6, va="bottom")
ax = fig.add_subplot(gs[0, 5])
for bb in BBN:
    mean_line(ax, G, KL[bb], C_BB[bb], marker="o", lw=1.1, ms=2.7, label=S_BB[bb])
ax.set_yscale("log"); ax.axhline(1.5, color=ST["Adapt"], lw=0.8, ls=(0, (3, 1.5)))
ax.text(1.0, 1.72, r"cap $\epsilon_D{=}1.5$", fontsize=5.8, color=ST["Adapt"])
ax.set_ylim(5e-3, 4); ax.set_xticks(G); ax.set_xlabel("round")
ax.set_ylabel(r"KL$(\hat p_w(d)\,\|\,\hat p(d))$", fontsize=6.6)
ax.legend(fontsize=5.4, loc="upper left", ncol=2, handlelength=1.1, columnspacing=0.6, bbox_to_anchor=(0.0, 0.87))
ptitle(ax, "b", "per-round topical drift")
save(fig, "app_gate_t3_isotropy")

# ====================================================================== 9. T4 route lattice
ROUTES_II = ["direct", "paraphrase", "expand", "contrast", "socratic"]
RLAB_II = {"direct": "direct", "paraphrase": "paraphrase", "expand": "expand", "contrast": "contrastive",
           "socratic": "Socratic"}
fig = plt.figure(figsize=(FULL, 2.25))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40, width_ratios=[1.45, 1, 1])
ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off()
TE = ["teacher_primary", "teacher_mistral"]; TLAB = {"teacher_primary": "Qwen2.5", "teacher_mistral": "Mistral"}
cnt = Counter()
for bb in BBN:
    for k, v in A[bb]["teacher_route"].items():
        cnt[tuple(k.split("|"))] += v
rate = {k: v / (320 * 4) for k, v in cnt.items()}
ty = {"teacher_primary": 0.69, "teacher_mistral": 0.31}
ry = {r: 0.91 - i * 0.205 for i, r in enumerate(ROUTES_II)}
rcm = LinearSegmentedColormap.from_list("acpb2", ["#C9DCEE", "#1F4E79"])
tot_r = Counter()
for (t, r), v in cnt.items():
    tot_r[r] += v
x0, x1 = 0.215, 0.735
for (t, r), v in sorted(cnt.items(), key=lambda kv: kv[1]):
    x = np.linspace(x0, x1, 80); y0, y1 = ty[t], ry[r]
    sx = (x - x0) / (x1 - x0)
    y = y0 + (y1 - y0) * (3 * sx ** 2 - 2 * sx ** 3)
    ax.plot(x, y, color=rcm((rate[(t, r)] - 0.18) / 0.47), lw=0.6 + v / 85, alpha=0.9, solid_capstyle="butt")
for t in TE:
    ax.add_patch(FancyBboxPatch((0.0, ty[t] - 0.085), 0.205, 0.17, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=ST["Synthesize"], ec="none"))
    ax.text(0.1025, ty[t] + 0.028, TLAB[t], color="white", ha="center", va="center", fontsize=6.2, fontweight="bold")
    ax.text(0.1025, ty[t] - 0.038, "teacher", color="white", ha="center", va="center", fontsize=5.2)
Hr = -sum((v / sum(tot_r.values())) * np.log2(v / sum(tot_r.values())) for v in tot_r.values())
for r in ROUTES_II:
    ax.add_patch(FancyBboxPatch((0.745, ry[r] - 0.068), 0.245, 0.136, boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc="white", ec=ST["Screen"], lw=0.6))
    ax.text(0.8675, ry[r] + 0.018, RLAB_II[r], ha="center", va="center", fontsize=6.0)
    ax.text(0.8675, ry[r] - 0.036, f"{100*tot_r[r]/sum(tot_r.values()):.0f}% of pool", ha="center", va="center",
            fontsize=5.0, color="#666")
ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.0)
ax.text(0.0, 1.035, r"$\mathbf{(a)}$ Regime II teacher → route lattice", fontsize=7.6, transform=ax.transAxes)
ax.text(0.0, -0.05, f"width $\\propto$ admitted samples; shade $\\propto$ admit rate (21–63%)\n"
        f"$H_{{\\mathrm{{route}}}}$ = {Hr:.2f} of 2.32 bits; $I(T;R)\\leq0.007$ bits", fontsize=5.6,
        transform=ax.transAxes, va="top", color="#444")

ROUTES_I = ["direct", "paraphrase", "expand", "contrast", "socratic"]
share = np.zeros((len(ROUTES_I), 8)); nset = 0
for m in models:
    for t in tasks:
        ok = True; S = np.zeros((len(ROUTES_I), 8))
        for g in range(1, 9):
            f = os.path.join(RAW, "runs", m, t, "acp", f"stats_round_{g}.json")
            if not os.path.exists(f): ok = False; break
            d = json.load(open(f)).get("accepted_route_dist")
            if not d: ok = False; break
            tot = sum(d.values())
            for i, r in enumerate(ROUTES_I):
                S[i, g - 1] = d.get(r, 0) / tot
        if ok:
            share += S; nset += 1
share /= max(nset, 1)
ax = fig.add_subplot(gs[0, 1])
rc = ["#1F4E79", "#6699CC", "#44AA99", "#DDAA33", "#AA3377"]
ax.stackplot(G, share * 100, colors=rc, labels=["direct", "paraphrase-first", "step-by-step", "contrast/verify",
                                                "Socratic"], lw=0.5, edgecolor="white")
ax.set_xlim(1, 8); ax.set_ylim(0, 100); ax.set_xticks(G); ax.set_xlabel("generation"); ax.grid(False)
ax.set_ylabel("share of admitted (%)")
ax.legend(fontsize=5.3, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, handlelength=0.8,
          columnspacing=0.5, handletextpad=0.3)
ptitle(ax, "b", f"Regime I route mix ({nset} settings)")

ax = fig.add_subplot(gs[0, 2])
for bb in BBN:
    pr = A[bb]["per_round_teacher"]
    q = [p.get("teacher_primary", 0) / max(1, sum(p.values())) * 100 for p in pr]
    mean_line(ax, G, q, C_BB[bb], marker="o", lw=1.1, ms=2.7, label=S_BB[bb])
ax.axhline(50, color="#555", lw=0.5, ls=(0, (2, 2)))
ax.set_ylim(30, 78); ax.set_xticks(G); ax.set_xlabel("round"); ax.set_ylabel("Qwen-teacher share (%)")
ax.legend(fontsize=5.3, loc="lower center", ncol=2, handlelength=1.0, columnspacing=0.5)
ptitle(ax, "c", "no teacher monopolises")
save(fig, "app_gate_t4_routes")

# ====================================================================== 10. Per-round screening funnel
fig, axs = plt.subplots(1, 2, figsize=(FULL, 1.85), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.18))
labs = ["consensus", "T1 band", "T2 floors", "dedup", "admitted"]
cols = [C_FUN["consensus"], C_FUN["T1"], C_FUN["T2"], C_FUN["dedup"], C_FUN["admitted"]]
ax = axs[0]
parts = [rcon.mean(0), r1.mean(0), r2.mean(0), rd.mean(0), fin.mean(0)]
bottom = np.zeros(8)
for p_, l_, c_ in zip(parts, labs, cols):
    ax.bar(G, p_ / 20, bottom=bottom, color=c_, width=0.72, label=l_, lw=0.4, edgecolor="white"); bottom += p_ / 20
ax.set_xticks(G); ax.set_ylim(0, 100); ax.set_xlabel("generation"); ax.set_ylabel("% of candidates")
ax.tick_params(axis="x", length=0)
ptitle(ax, "a", "Regime I (mean of 16 settings, 2,000 candidates/round)")
ax = axs[1]
diag = {bb: A[bb]["diag"] for bb in BBN}


def mean_key(key):
    return np.array([[d[key] for d in diag[bb]] for bb in BBN]).mean(0)


gen = mean_key("num_generated"); inp = mean_key("input_candidates")
parts = [gen - inp, mean_key("reject_t1"), mean_key("reject_t2"), mean_key("reject_commit_dup"), mean_key("num_accepted")]
bottom = np.zeros(8)
for p_, l_, c_ in zip(parts, labs, cols):
    ax.bar(G, 100 * p_ / gen, bottom=bottom, color=c_, width=0.72, lw=0.4, edgecolor="white"); bottom += 100 * p_ / gen
ax.set_xticks(G); ax.set_ylim(0, 100); ax.set_xlabel("round"); ax.tick_params(axis="x", length=0)
ptitle(ax, "b", "Regime II (mean of 4 backbones, ≈400 candidates/round)")
fig.legend(*axs[0].get_legend_handles_labels(), fontsize=6.0, ncol=5, loc="upper center",
           bbox_to_anchor=(0.5, -0.02), handlelength=0.9, columnspacing=1.0)
save(fig, "app_screening_per_round")

# ====================================================================== 11. Accuracy trajectories (4x4)
fig, axs = plt.subplots(4, 4, figsize=(FULL, 5.4), sharex=True)
for i, m in enumerate(models):
    for j, t in enumerate(tasks):
        ax = axs[i, j]; v = B[f"{m}|{t}"]; a = np.array(v["acc"]) * 100; n = v["n"]
        se = 196 * np.sqrt(a / 100 * (1 - a / 100) / n)
        ax.fill_between(R, a - se, a + se, color=C_MODEL[m], alpha=0.14, lw=0)
        ax.axhline(a[0], color="#555", lw=0.5, ls=(0, (2, 2)))
        mean_line(ax, R, a, C_MODEL[m], marker=M_TASK[t], lw=1.1, ms=2.8, label="ACP")
        if m == "mistral-7b" and t == "gsm8k":
            mean_line(ax, range(5), np.array(NV["acc"]) * 100, C_NAIVE, marker="o", lw=1.0, ms=2.6, label="naive")
            ax.legend(fontsize=5.8, loc="lower right", ncol=2, columnspacing=0.6)
        ax.set_title(f"{N_MODEL[m].split('-Instruct')[0].split('-it')[0]} / {N_TASK[t]}  "
                     f"($\\Delta$={100*v['delta']:+.1f})", fontsize=6.8, loc="left")
        ax.set_xticks(R)
        if i == 3: ax.set_xlabel("generation")
        if j == 0: ax.set_ylabel("accuracy (%)")
fig.tight_layout(h_pad=0.7, w_pad=0.7)
save(fig, "app_accuracy_trajectories")

# ====================================================================== 12. Regime II per-round metrics (4x5)
keys5 = [("tail_mass", "tail mass (%)"), ("entropy_mean", r"mean $H_{\pi_0}$ (bits)"), ("entropy_std", r"s.d. $H_{\pi_0}$"),
         ("repetition", "bigram repetition"), ("heaps", r"Heaps $\beta$ (adm.: cumulative)")]
fig, axs = plt.subplots(4, 5, figsize=(FULL, 4.6), sharex=True)
for i, b in enumerate(bbs):
    for j, (k, lab) in enumerate(keys5):
        ax = axs[i, j]; g = A[b]["ml"]["generated"][k]; a = A[b]["ml"]["accepted"][k]
        mean_line(ax, G, g, C_GEN, marker="o", lw=0.9, ms=2.4, label="generated")
        mean_line(ax, G, a, C_BB[b], marker="s", lw=1.1, ms=2.4, label="admitted")
        if i == 0: ax.set_title(lab, fontsize=6.8)
        if j == 0: ax.set_ylabel(S_BB[b], fontsize=6.8, fontweight="bold", color=C_BB[b])
        if i == 3: ax.set_xlabel("round")
        ax.set_xticks(G); ax.tick_params(labelsize=5.7)
fig.legend(handles=[Line2D([], [], color=C_GEN, marker="o", ms=2.6, lw=0.9, label="generated"),
                    Line2D([], [], color="#333", marker="s", ms=2.6, lw=1.1, label="admitted (backbone colour)")],
           fontsize=6.2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(h_pad=0.5, w_pad=0.5)
save(fig, "app_openweb_per_round")

# ====================================================================== 13. Route entropy + counterfactual probes
fig, axs = plt.subplots(1, 2, figsize=(FULL, 1.85), gridspec_kw={"wspace": 0.25})
ax = axs[0]
for m in models:
    for t in tasks:
        v = B[f"{m}|{t}"]
        ax.plot(G, v["route_H"], color=C_MODEL[m], marker=M_TASK[t], ms=2.1, lw=0.6, alpha=0.75, mew=0)
for b in bbs:
    ax.plot(G, [x["final_route_entropy"] for x in A[b]["diag"]], color=C_BB[b], ls=(0, (3, 1.5)), lw=0.9)
ax.axhline(np.log2(5), color="#333", lw=0.6, ls=(0, (1, 1.5)))
ax.text(8, np.log2(5) + 0.06, r"max $\log_2 5$", fontsize=5.8, ha="right")
ax.axhline(1.0, color=ST["Adapt"], lw=0.8); ax.set_ylim(0.9, 2.55)
ax.text(1, 1.04, "tripwire floor (1 bit)", fontsize=5.8, color=ST["Adapt"])
ax.set_xlabel("round"); ax.set_ylabel("route entropy of admitted (bits)"); ax.set_xticks(G)
ptitle(ax, "a", "route entropy (solid: Regime I; dashed: Regime II)")
ax = axs[1]
for m in models:
    for t in tasks:
        v = B[f"{m}|{t}"]
        ax.plot(G, v["cf"], color=C_MODEL[m], marker=M_TASK[t], ms=2.1, lw=0.6, alpha=0.75, mew=0)
for b in bbs:
    ax.plot(G, [x["cf_catch_rate"] for x in A[b]["diag"]], color=C_BB[b], ls=(0, (3, 1.5)), lw=0.9)
ax.set_xlabel("round"); ax.set_ylabel("counterfactual catch rate"); ax.set_ylim(0.3, 0.88); ax.set_xticks(G)
h = [Line2D([], [], color=C_MODEL[m], lw=1.2, label=S_MODEL[m]) for m in models]
ax.legend(handles=h, fontsize=5.6, ncol=4, loc="upper left", handlelength=1.1, columnspacing=0.7)
ptitle(ax, "b", "gates/critics reject perturbed probes")
save(fig, "app_route_entropy_probes")

# ====================================================================== 14. Band drift + training loss
fig, axs = plt.subplots(1, 2, figsize=(FULL, 1.85), gridspec_kw={"wspace": 0.25})
ax = axs[0]
for m in models:
    for t in tasks:
        v = B[f"{m}|{t}"]; lo = [b[0] for b in v["band"]]; hi = [b[1] for b in v["band"]]
        ax.plot(G, [(a + b) / 2 for a, b in zip(lo, hi)], color=C_MODEL[m], marker=M_TASK[t], ms=2.3, lw=0.8, mew=0)
ax.set_xlabel("round"); ax.set_ylabel(r"band centre of $H_{\pi_0}$ (bits/token)"); ax.set_xticks(G)
h = model_task_legend(ax)
ax.legend(handles=h, fontsize=5.4, ncol=4, loc="upper left", handlelength=1.0, columnspacing=0.5, handletextpad=0.25)
ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
ptitle(ax, "a", r"Regime I: self-generated text drifts from $\pi_0$")
ax = axs[1]
for m in models:
    for t in tasks:
        v = B[f"{m}|{t}"]
        ax.plot(G, v["loss"], color=C_MODEL[m], marker=M_TASK[t], ms=2.3, lw=0.8, mew=0)
ax.plot(range(1, 5), [s["train_loss_mean"] for s in NV["stats"]], color=C_NAIVE, lw=1.5, ls=(0, (3, 1.5)),
        label="naive (Mistral/GSM8K)")
ax.legend(fontsize=6.0); ax.set_xlabel("round"); ax.set_ylabel("mean LoRA training loss"); ax.set_xticks(G)
ptitle(ax, "b", "training loss on the cumulative pool")
save(fig, "app_drift_and_loss")
print("done")
