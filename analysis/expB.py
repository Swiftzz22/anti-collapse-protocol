import json, os, numpy as np
from scipy.stats import binomtest
import os
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","raw_logs","runs")
models=["gemma2-9b","llama3.1-8b","mistral-7b","qwen2.5-7b"]; tasks=["gsm8k","arc_challenge","commonsense_qa","boolq"]
def ev(m,t,c,s):
    p=f"{R}/{m}/{t}/_base/eval_step_0.json" if s==0 else f"{R}/{m}/{t}/{c}/eval_step_{s}.json"
    return json.load(open(p))
out={}
for m in models:
  for t in tasks:
    accs=[];corr=[]
    for s in range(9):
        accs.append(json.load(open(f"{R}/{m}/{t}/acp/results.json"))["accuracy"][str(s)])
    n={"gsm8k":1319,"arc_challenge":1172,"commonsense_qa":1221,"boolq":3270}[t]
    from scipy.stats import norm
    def ztest(p1,p2):
        pp=(p1+p2)/2; se=np.sqrt(2*pp*(1-pp)/n); z=(p2-p1)/se; return 2*(1-norm.cdf(abs(z)))
    p=ztest(accs[0],accs[8]); b=c=None
    worst=int(np.argmin(accs[1:]))+1
    pw=ztest(accs[0],accs[worst])
    st=[json.load(open(f"{R}/{m}/{t}/acp/stats_round_{r}.json")) for r in range(1,9)]
    out[f"{m}|{t}"]=dict(n=n,acc=accs,delta=accs[8]-accs[0],mcn_p=p,b=b,c=c,worst=worst,worst_acc=accs[worst],worst_p=pw,
      gen_prec=[s["generated_label_precision"] for s in st],acc_prec=[s["accepted_label_precision"] for s in st],
      accept_rate=[s["accept_rate"] for s in st],after_corr=[s["num_after_correctness"] for s in st],
      rej_t1=[s.get("reject_t1",0) for s in st],rej_t2=[s.get("reject_t2",0) for s in st],rej_t3=[s.get("reject_t3",0) for s in st],
      rej_dup=[s.get("reject_commit_dup",0) for s in st],rej_crit=[s.get("reject_critics",0) for s in st],rej_route=[s.get("reject_route_diversity",0)+s.get("reject_global_route_balance",0) for s in st],
      final=[s["final_accepted"] for s in st],route_H=[s["final_route_entropy"] for s in st],
      cf=[s.get("cf_catch_rate") for s in st],band=[s["entropy_band"] for s in st],pool=[s["pool_size"] for s in st],
      loss=[s.get("train_loss_mean") for s in st],forced=[("forced_critics" in s) for s in st],critics=[s["active_critics"] for s in st])
json.dump(out,open("expB.json","w"),indent=1)
print(f"{'cell':28s} {'S0':>6} {'S8':>6} {'Δ':>6} {'p':>6} {'min':>6} {'pw':>6} genP accP acc% routeH cf")
D=[];AD=[];WD=[]
for k,v in out.items():
    D.append(v['delta']);WD.append(min(v['acc'][1:])-v['acc'][0])
    print(f"{k:28s} {100*v['acc'][0]:6.1f} {100*v['acc'][8]:6.1f} {100*v['delta']:+6.1f} {v['mcn_p']:6.3f} {100*v['worst_acc']:6.1f} {v['worst_p']:6.3f} {np.mean(v['gen_prec']):.3f} {np.mean(v['acc_prec']):.3f} {100*np.mean(v['accept_rate']):.1f} {np.mean(v['route_H']):.3f} {np.mean(v['cf']):.3f}")
D=np.array(D)*100
print("mean Δ",D.mean(),"median",np.median(D),"mean|Δ|",np.abs(D).mean(),"min",D.min(),"max",D.max())
print("n p<0.05 S8:",sum(v['mcn_p']<0.05 for v in out.values()), [k for k,v in out.items() if v['mcn_p']<0.05])
print("worst-round drop mean",np.mean(WD)*100, "min", np.min(WD)*100)
# Bonferroni
print("Bonferroni-sig:", [k for k,v in out.items() if v['mcn_p']<0.05/12])
