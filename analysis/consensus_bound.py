"""Prop. 1 bound, common-mode estimate rho-hat (Eq. 5). Reads expB.json; writes bound.json, rho.json."""
import json, numpy as np
from scipy.stats import binom
K, m = 5, 3
L = {"gsm8k": None, "arc_challenge": 3, "commonsense_qa": 4, "boolq": 1}
o = json.load(open("expB.json")); rows = []; rho = []
for k, v in o.items():
    t = k.split("|")[1]; p = np.mean(v["gen_prec"]); obs = np.mean(v["acc_prec"])
    A = binom.sf(m - 1, K, p)
    B = 0.0 if L[t] is None else min(1.0, L[t] * binom.sf(m - 1, K, (1 - p) / L[t]))
    pib = A / (A + B)
    r = (A - obs * (A + B)) / (obs - obs * (A + B) - p + A)
    rows.append((k, p, obs, pib, A, None)); rho.append((k, r))
    print(f"{k:28s} p={p:.3f} obs={obs:.3f} bound={pib:.3f} rho={r:.2f}")
json.dump(rows, open("bound.json", "w")); json.dump(rho, open("rho.json", "w"))
