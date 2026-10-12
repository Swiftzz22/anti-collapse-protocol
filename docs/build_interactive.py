"""Build the script-free interactive sections of docs/index.html from raw_logs/.

The anonymous review viewer blocks JavaScript by default, so all interactivity is pure
HTML/CSS: radio buttons select which pre-rendered panel is visible, and SVG <title>
elements give hover tooltips. Re-run after the logs change:

    python docs/build_interactive.py
"""
import json
import math
import os
import re
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "raw_logs", "runs")
INDEX = os.path.join(HERE, "index.html")

MODELS = [("gemma2-9b", "Gemma-2-9B"), ("llama3.1-8b", "Llama-3.1-8B"),
          ("mistral-7b", "Mistral-7B"), ("qwen2.5-7b", "Qwen2.5-7B")]


def sid(key):
    """Selector-safe id: a '.' inside an id would invalidate the whole CSS selector list."""
    return key.replace(".", "_")
TASKS = [("gsm8k", "GSM8K"), ("arc_challenge", "ARC-C"),
         ("commonsense_qa", "CSQA"), ("boolq", "BoolQ")]
N_TEST = {"gsm8k": 1319, "arc_challenge": 1172, "commonsense_qa": 1221, "boolq": 3270}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(model, task, cond="acp"):
    d = os.path.join(RUNS, model, task, cond)
    if not os.path.isdir(d):
        return None
    res = load(os.path.join(d, "results.json"))
    acc = [res["accuracy"][str(s)] for s in range(len(res["accuracy"]))]
    rounds = sorted(int(k) for k in res["rounds"])
    st = [load(os.path.join(d, f"stats_round_{r}.json")) for r in rounds]
    if cond == "naive":
        # results.json for the naive run stops at G3; analysis/naive.json holds the full G0-G4 used in the paper
        full = load(os.path.join(HERE, "..", "analysis", "naive.json"))
        acc, st = full["acc"], full["stats"]
    return dict(acc=acc, gen=[s["generated_label_precision"] for s in st],
                adm=[s["accepted_label_precision"] for s in st],
                rate=[s["accept_rate"] for s in st], n=[s["num_accepted"] for s in st])


def pool_wrong(r, rounds):
    """Share of wrong labels in the cumulative training pool after `rounds` rounds."""
    n = r["n"][:rounds]
    return sum(k * (1 - a) for k, a in zip(n, r["adm"])) / sum(n)


def nice_ticks(lo, hi, target=5):
    span = hi - lo
    step = min((s for s in (0.5, 1, 2, 5, 10, 20) if span / s <= target), default=20)
    t0 = math.floor(lo / step) * step
    t1 = math.ceil(hi / step) * step
    ticks, v = [], t0
    while v <= t1 + 1e-9:
        ticks.append(round(v, 2))
        v += step
    return ticks


def signed(v):
    """+1.2 / −1.2 with a typographic minus."""
    return f"{v:+.1f}".replace("-", "−")


def fmt(v):
    return f"{v:g}"


class Chart:
    """A minimal SVG line chart; colours come from CSS classes so both themes work."""
    W, H, L, R, T, B = 440, 230, 44, 14, 16, 34

    def __init__(self, xs, ylo, yhi, xlabel, ylabel):
        self.xs = xs
        self.ticks = nice_ticks(ylo, yhi)
        self.ylo, self.yhi = self.ticks[0], self.ticks[-1]
        self.parts = []
        self.xlabel, self.ylabel = xlabel, ylabel

    def x(self, v):
        a, b = self.xs[0], self.xs[-1]
        return self.L + (v - a) / (b - a) * (self.W - self.L - self.R)

    def y(self, v):
        return self.T + (self.yhi - v) / (self.yhi - self.ylo) * (self.H - self.T - self.B)

    def axes(self):
        out = []
        for t in self.ticks:
            yy = self.y(t)
            out.append(f'<line class="grid" x1="{self.L}" x2="{self.W - self.R}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
            out.append(f'<text class="tick" x="{self.L - 6}" y="{yy + 3.5:.1f}" text-anchor="end">{fmt(t)}</text>')
        for v in self.xs:
            out.append(f'<text class="tick" x="{self.x(v):.1f}" y="{self.H - self.B + 15}" text-anchor="middle">{v}</text>')
        out.append(f'<text class="axl" x="{(self.L + self.W - self.R) / 2:.1f}" y="{self.H - 3}" text-anchor="middle">{self.xlabel}</text>')
        out.append(f'<text class="axl" transform="translate(11 {(self.T + self.H - self.B) / 2:.1f}) rotate(-90)" text-anchor="middle">{self.ylabel}</text>')
        return out

    def line(self, xs, ys, cls, tip):
        pts = " ".join(f"{self.x(a):.1f},{self.y(b):.1f}" for a, b in zip(xs, ys))
        self.parts.append(f'<polyline class="{cls}" points="{pts}"/>')
        for a, b in zip(xs, ys):
            self.parts.append(f'<circle class="{cls} pt" cx="{self.x(a):.1f}" cy="{self.y(b):.1f}" r="3.6">'
                              f'<title>{escape(tip(a, b))}</title></circle>')

    def area(self, xs, top, bottom, cls):
        pts = [f"{self.x(a):.1f},{self.y(b):.1f}" for a, b in zip(xs, top)]
        pts += [f"{self.x(a):.1f},{self.y(b):.1f}" for a, b in reversed(list(zip(xs, bottom)))]
        self.parts.insert(0, f'<polygon class="{cls}" points="{" ".join(pts)}"/>')

    def band(self, lo, hi, cls):
        self.parts.insert(0, f'<rect class="{cls}" x="{self.L}" y="{self.y(hi):.1f}" '
                             f'width="{self.W - self.L - self.R}" height="{self.y(lo) - self.y(hi):.1f}"/>')

    def hline(self, v, cls):
        self.parts.insert(0, f'<line class="{cls}" x1="{self.L}" x2="{self.W - self.R}" y1="{self.y(v):.1f}" y2="{self.y(v):.1f}"/>')

    def svg(self, label):
        body = "".join(self.axes() + self.parts)
        return (f'<svg viewBox="0 0 {self.W} {self.H}" role="img" aria-label="{escape(label)}">'
                f'{body}</svg>')


def legend(items):
    return '<div class="xleg">' + "".join(f'<span class="lg {c}">{escape(t)}</span>' for c, t in items) + "</div>"


def panel(mk, mname, tk, tname):
    r = run(mk, tk)
    naive = run(mk, tk, "naive")
    acc = [100 * a for a in r["acc"]]
    base = acc[0]
    n = N_TEST[tk]
    ci = 196 * math.sqrt(r["acc"][0] * (1 - r["acc"][0]) / n)
    delta = acc[8] - base

    # Accuracy chart
    vals = acc + ([100 * a for a in naive["acc"]] if naive else []) + [base - ci, base + ci]
    ca = Chart(list(range(9)), min(vals) - 0.4, max(vals) + 0.4, "self-consumption generation", "accuracy (%)")
    ca.band(base - ci, base + ci, "ci")
    ca.hline(base, "base")
    if naive:
        nacc = [100 * a for a in naive["acc"]]
        ca.line(list(range(len(nacc))), nacc, "naive", lambda a, b: f"Naive loop, G{a}: {b:.1f}%")
    ca.line(list(range(9)), acc, "acp", lambda a, b: f"ACP, G{a}: {b:.1f}%")
    leg_a = [("acp", "ACP"), ("base", "base model"), ("ci", "base 95% CI")]
    if naive:
        leg_a.insert(1, ("naive", "naive loop (no filtering)"))

    # Precision chart
    gen = [100 * v for v in r["gen"]]
    adm = [100 * v for v in r["adm"]]
    rounds = list(range(1, 9))
    pvals = gen + adm + ([100 * v for v in naive["adm"]] if naive else [])
    cp = Chart(rounds, min(pvals) - 1, max(pvals) + 1, "round", "label precision (%)")
    cp.area(rounds, adm, gen, "gap")
    if naive:
        nadm = [100 * v for v in naive["adm"]]
        cp.line(list(range(1, len(nadm) + 1)), nadm, "naive", lambda a, b: f"Naive pool, round {a}: {b:.1f}% correct")
    cp.line(rounds, gen, "gen", lambda a, b: f"Generated, round {a}: {b:.1f}% correct")
    cp.line(rounds, adm, "acp", lambda a, b: f"Admitted by ACP, round {a}: {b:.1f}% correct")
    leg_p = [("acp", "admitted by ACP"), ("gen", "generated by the student")]
    if naive:
        leg_p.append(("naive", "naive pool"))

    gm, am = sum(gen) / 8, sum(adm) / 8
    noise_cut = 100 * (1 - (100 - am) / (100 - gm))
    rate = 100 * sum(r["rate"]) / 8
    within = abs(delta) <= ci
    dcls = "pos" if delta > 0.05 else ("neg" if delta < -0.05 else "")

    stats = [
        (f"{base:.1f}%", "base accuracy (G0)"),
        (f'<span class="{dcls}">{signed(delta)} pp</span>', f"after 8 generations ({'within' if within else 'outside'} ±{ci:.1f} pp)"),
        (f"{gm:.1f}→{am:.1f}%", "label precision, generated → admitted"),
        (f"{noise_cut:.0f}%", "fewer wrong labels in training"),
        (f"{rate:.0f}%", "of candidates admitted per round"),
    ]
    stat_html = "".join(f'<div class="xs"><div class="n">{v}</div><div class="l">{escape(l)}</div></div>' for v, l in stats)

    note = (f"The student's own outputs went from {gen[0]:.1f}% correct in round 1 to {gen[-1]:.1f}% in round 8; "
            f"the data ACP admitted stayed between {min(adm):.1f}% and {max(adm):.1f}%.")
    if naive:
        g = len(naive["acc"]) - 1
        note += (f" The naive loop trains on everything: by generation {g} its training pool is "
                 f"{100 * pool_wrong(naive, g):.1f}% wrong versus {100 * pool_wrong(r, g):.1f}% for ACP, yet its accuracy "
                 f"({100 * naive['acc'][g]:.1f}%) matches ACP's ({acc[g]:.1f}%).")

    return (f'<div class="xp p-{sid(mk)} p-{tk}" role="group" aria-label="{escape(mname)} on {escape(tname)}">'
            f'<div class="xstats">{stat_html}</div>'
            f'<div class="xcharts">'
            f'<div class="xc"><div class="xct">Test accuracy</div>{ca.svg(f"{mname} {tname} accuracy over 8 generations")}{legend(leg_a)}</div>'
            f'<div class="xc"><div class="xct">Training-label precision</div>{cp.svg(f"{mname} {tname} label precision by round")}{legend(leg_p)}</div>'
            f'</div><p class="xnote">{escape(note)}</p></div>')


def explorer():
    radios = "".join(f'<input type="radio" name="xm" id="xm-{sid(k)}" class="vh"{" checked" if i == 0 else ""}>'
                     for i, (k, _) in enumerate(MODELS))
    radios += "".join(f'<input type="radio" name="xt" id="xt-{k}" class="vh"{" checked" if i == 0 else ""}>'
                      for i, (k, _) in enumerate(TASKS))
    seg_m = "".join(f'<label for="xm-{sid(k)}">{escape(n)}</label>' for k, n in MODELS)
    seg_t = "".join(f'<label for="xt-{k}">{escape(n)}</label>' for k, n in TASKS)
    panels = "".join(panel(mk, mn, tk, tn) for mk, mn in MODELS for tk, tn in TASKS)
    return (f'<section class="explorer" aria-labelledby="explore">\n'
            f'  <h2 id="explore">Explore every run</h2>\n'
            f'  <p class="hint">Pick a student and a task. Hover a point for its exact value. Mistral-7B + GSM8K also shows the unfiltered (naive) loop.</p>\n'
            f'  {radios}\n'
            f'  <div class="xctl"><div class="seg"><span>Student</span>{seg_m}</div>'
            f'<div class="seg"><span>Task</span>{seg_t}</div></div>\n'
            f'  <div class="xpanels">{panels}</div>\n'
            f'</section>')


def explorer_css():
    rules = []
    for mk, _ in MODELS:
        rules.append(f"#xm-{sid(mk)}:checked~.xctl label[for=xm-{sid(mk)}]")
    for tk, _ in TASKS:
        rules.append(f"#xt-{tk}:checked~.xctl label[for=xt-{tk}]")
    css = [",".join(rules) + "{background:var(--ink);color:var(--bg);border-color:var(--ink)}"]
    foc = [f"#xm-{sid(k)}:focus-visible~.xctl label[for=xm-{sid(k)}]" for k, _ in MODELS]
    foc += [f"#xt-{k}:focus-visible~.xctl label[for=xt-{k}]" for k, _ in TASKS]
    css.append(",".join(foc) + "{outline:2px solid var(--accent);outline-offset:2px}")
    show = [f"#xm-{sid(mk)}:checked~#xt-{tk}:checked~.xpanels .p-{sid(mk)}.p-{tk}" for mk, _ in MODELS for tk, _ in TASKS]
    css.append(",".join(show) + "{display:block}")
    return "\n".join(css)


# ------------------------------------------------------------------ consensus calculator
K, M = 5, 3
P_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]
L_VALUES = [("1", 1, "yes / no"), ("3", 3, "4-way choice"), ("inf", math.inf, "free-form number")]
RHO_VALUES = [0.0, 0.25, 0.5, 0.75]


def fbar(m, q):
    return sum(math.comb(K, j) * q ** j * (1 - q) ** (K - j) for j in range(m, K + 1))


def consensus(p, L, rho):
    A = fbar(M, p)
    B = 0.0 if math.isinf(L) else L * fbar(M, (1 - p) / L)
    prec = (rho * p + (1 - rho) * A) / (rho + (1 - rho) * (A + B))
    reach = rho + (1 - rho) * (A + B)
    return prec, reach


def bound():
    radios = "".join(f'<input type="radio" name="bp" id="bp-{int(p * 100)}" class="vh"{" checked" if p == 0.6 else ""}>' for p in P_VALUES)
    radios += "".join(f'<input type="radio" name="bl" id="bl-{k}" class="vh"{" checked" if k == "3" else ""}>' for k, _, _ in L_VALUES)
    radios += "".join(f'<input type="radio" name="br" id="br-{int(r * 100)}" class="vh"{" checked" if r == 0 else ""}>' for r in RHO_VALUES)
    seg_p = "".join(f'<label for="bp-{int(p * 100)}">{int(p * 100)}%</label>' for p in P_VALUES)
    seg_l = "".join(f'<label for="bl-{k}">{escape(n)}</label>' for k, _, n in L_VALUES)
    seg_r = "".join(f'<label for="br-{int(r * 100)}">{r:g}</label>' for r in RHO_VALUES)
    outs = []
    for p in P_VALUES:
        for k, L, _ in L_VALUES:
            for r in RHO_VALUES:
                prec, reach = consensus(p, L, r)
                cut = 100 * (1 - (1 - prec) / (1 - p))
                outs.append(
                    f'<div class="bo o-{int(p * 100)}-{k}-{int(r * 100)}">'
                    f'<div class="bars"><div class="bar"><span>generated</span><i style="width:{100 * p:.1f}%"></i><b>{100 * p:.1f}%</b></div>'
                    f'<div class="bar acp"><span>admitted</span><i style="width:{100 * prec:.1f}%"></i><b>{100 * prec:.1f}%</b></div></div>'
                    f'<div class="bstats"><div><b>{signed(100 * (prec - p))} pp</b><span>precision lift</span></div>'
                    f'<div><b>{cut:.0f}%</b><span>fewer wrong labels</span></div>'
                    f'<div><b>{100 * reach:.0f}%</b><span>of prompts reach consensus</span></div></div></div>')
    return (f'<div class="calc" aria-labelledby="calc-h">\n'
            f'  <h3 id="calc-h">Try it: what does consensus buy you?</h3>\n'
            f'  <p class="hint">{M}-of-{K} consensus. ρ is the share of prompts where every route makes the same mistake '
            f'(observed ρ̂: GSM8K 0.09–0.38, ARC-C 0.36–0.66, CSQA 0.64–0.80).</p>\n'
            f'  {radios}\n'
            f'  <div class="xctl"><div class="seg"><span>Route accuracy <i>p</i></span>{seg_p}</div>'
            f'<div class="seg"><span>Answer type</span>{seg_l}</div>'
            f'<div class="seg"><span>Shared errors ρ</span>{seg_r}</div></div>\n'
            f'  <div class="bouts" aria-live="polite">{"".join(outs)}</div>\n'
            f'</div>')


def bound_css():
    act = [f"#bp-{int(p * 100)}:checked~.xctl label[for=bp-{int(p * 100)}]" for p in P_VALUES]
    act += [f"#bl-{k}:checked~.xctl label[for=bl-{k}]" for k, _, _ in L_VALUES]
    act += [f"#br-{int(r * 100)}:checked~.xctl label[for=br-{int(r * 100)}]" for r in RHO_VALUES]
    foc = [a.replace(":checked", ":focus-visible") for a in act]
    show = [f"#bp-{int(p * 100)}:checked~#bl-{k}:checked~#br-{int(r * 100)}:checked~.bouts .o-{int(p * 100)}-{k}-{int(r * 100)}"
            for p in P_VALUES for k, _, _ in L_VALUES for r in RHO_VALUES]
    return "\n".join([",".join(act) + "{background:var(--ink);color:var(--bg);border-color:var(--ink)}",
                      ",".join(foc) + "{outline:2px solid var(--accent);outline-offset:2px}",
                      ",".join(show) + "{display:block}"])


STATIC_CSS = """
.explorer,.calc{margin:2.4rem 0;position:relative}
.calc{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:1rem 1.1rem 1.2rem}
.calc h3{margin-top:.2rem}
.xctl{display:flex;flex-wrap:wrap;gap:.6rem 1.4rem;margin:1rem 0}
.seg{display:flex;flex-wrap:wrap;align-items:center;gap:.35rem;font:500 .84rem Inter,sans-serif}
.seg span{color:var(--muted);margin-right:.2rem;font-size:.76rem;text-transform:uppercase;letter-spacing:.06em}
.seg label{cursor:pointer;border:1px solid var(--line);border-radius:999px;padding:.32rem .8rem;background:var(--surface);color:var(--ink);user-select:none}
.seg label:hover{border-color:var(--muted)}
.xp,.bo{display:none}
.xp{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:1rem}
.xstats{display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem;margin-bottom:.8rem}
.xs .n{font:600 1.15rem Inter,sans-serif;letter-spacing:-.01em}
.xs .l{font:400 .74rem/1.35 Inter,sans-serif;color:var(--muted)}
.xcharts{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.xct{font:600 .82rem Inter,sans-serif;margin-bottom:.2rem}
.xc svg{width:100%;height:auto;display:block;overflow:visible}
.xnote{font:400 .86rem/1.55 Inter,sans-serif;color:var(--muted);margin:.8rem 0 0}
.xleg{display:flex;flex-wrap:wrap;gap:.3rem .9rem;font:400 .74rem Inter,sans-serif;color:var(--muted)}
.lg::before{content:"";display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:.35rem;vertical-align:middle;background:var(--muted)}
.lg.acp::before{background:var(--accent)} .lg.naive::before{background:var(--bad)}
.lg.ci::before{height:9px;background:var(--line)} .lg.base::before{background:repeating-linear-gradient(90deg,var(--muted) 0 4px,transparent 4px 7px)}
svg .grid{stroke:var(--line);stroke-width:1}
svg .tick{fill:var(--muted);font:400 10.5px Inter,sans-serif}
svg .axl{fill:var(--muted);font:500 11px Inter,sans-serif}
svg .ci{fill:var(--line);opacity:.7}
svg .base{stroke:var(--muted);stroke-dasharray:4 3;stroke-width:1.2}
svg .gap{fill:var(--accent);opacity:.12}
svg polyline{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}
svg polyline.acp{stroke:var(--accent)} svg polyline.gen{stroke:var(--muted)} svg polyline.naive{stroke:var(--bad);stroke-dasharray:5 3}
svg .pt{stroke:var(--surface);stroke-width:1.5;cursor:help}
svg .pt.acp{fill:var(--accent)} svg .pt.gen{fill:var(--muted)} svg .pt.naive{fill:var(--bad)}
svg .pt:hover{r:6}
.bars{display:grid;gap:.45rem;margin:.4rem 0 1rem}
.bar{display:grid;grid-template-columns:5.5rem 1fr 3.6rem;align-items:center;gap:.6rem;font:400 .82rem Inter,sans-serif;color:var(--muted)}
.bar i{display:block;height:14px;border-radius:3px;background:var(--muted);opacity:.55}
.bar.acp i{background:var(--accent);opacity:1}
.bar b{font-weight:600;color:var(--ink);text-align:right;font-variant-numeric:tabular-nums}
.bstats{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem}
.bstats b{display:block;font:600 1.15rem Inter,sans-serif}
.bstats span{font:400 .76rem Inter,sans-serif;color:var(--muted)}
@media (max-width:720px){.xstats{grid-template-columns:repeat(2,1fr)}.xcharts{grid-template-columns:1fr}.xctl{gap:.6rem}}
"""


def splice(html, start, end, content):
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pat.search(html):
        raise SystemExit(f"marker {start!r} not found in index.html")
    return pat.sub(lambda _: f"{start}\n{content}\n{end}", html, count=1)


def main():
    with open(INDEX, encoding="utf-8") as f:
        html = f.read()
    html = splice(html, "/* GEN:CSS */", "/* /GEN:CSS */", STATIC_CSS.strip() + "\n" + explorer_css() + "\n" + bound_css())
    html = splice(html, "<!-- GEN:EXPLORER -->", "<!-- /GEN:EXPLORER -->", explorer())
    html = splice(html, "<!-- GEN:BOUND -->", "<!-- /GEN:BOUND -->", bound())
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(html)
    print("updated", INDEX)


if __name__ == "__main__":
    main()
