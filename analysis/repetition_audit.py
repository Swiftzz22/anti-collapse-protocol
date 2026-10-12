"""Size-matched repetition audit (App. D) for the Qwen2.5-7B open-web run. Requires `pip install tokenizers`."""
import json, os, random, numpy as np
from collections import Counter
from tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("Qwen/Qwen2.5-7B")
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw_logs", "openweb_qwen")
acc = [json.loads(l) for l in open(f"{B}/accepted.jsonl")]
ml = json.load(open(f"{B}/metrics_log.json"))
GEN_N = [90332, 88839, 89040, 88794, 88146, 89565, 88795, 87529]   # generated-pool token counts from the run log
def rep(docs):
    t = [x for d in docs for x in d]; c = Counter(zip(t, t[1:]))
    return sum(v - 1 for v in c.values() if v > 1) / (len(t) - 1), len(t)
rng = random.Random(1); res = []
for r in range(1, 9):
    docs = [tok.encode(a["output"]).ids for a in acc if a["round_id"] == r]; pts = []
    for f in [0.25, 0.5, 0.75, 1.0]:
        vs = [rep(rng.sample(docs, max(2, int(len(docs) * f)))) for _ in range(30 if f < 1 else 1)]
        pts.append((np.mean([v[1] for v in vs]), np.mean([v[0] for v in vs])))
    a, b = np.polyfit(np.log([p[0] for p in pts]), [p[1] for p in pts], 1)
    res.append((r, pts[-1][1], a * np.log(GEN_N[r - 1]) + b, ml["generated"]["repetition"][r - 1]))
    print(res[-1])
json.dump(res, open("rep_sizecheck.json", "w"))
