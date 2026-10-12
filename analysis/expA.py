import json,csv,numpy as np,glob
from collections import Counter
import os
B=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","raw_logs")
runs={"Gemma-2-9B":"openweb_gemma","Llama-3-8B":"openweb_llama","Mistral-7B-v0.3":"openweb_mistral","Qwen2.5-7B":"openweb_qwen"}
out={}
for name,d in runs.items():
    base=f"{B}/{d}"
    ml=json.load(open(f"{base}/metrics_log.json"))
    diag=json.load(open(f"{base}/t4_diagnostics.json"))
    acc=[json.loads(l) for l in open(f"{base}/accepted.jsonl")]
    tr=Counter((a['teacher_name'],a['route']) for a in acc)
    te=Counter(a['teacher_name'] for a in acc)
    out[name]=dict(ml=ml,diag=diag,teacher=dict(te),teacher_route={f"{k[0]}|{k[1]}":v for k,v in tr.items()},
                   per_round_teacher=[dict(Counter(a['teacher_name'] for a in acc if a['round_id']==r)) for r in range(1,9)],
                   n_acc=len(acc))
json.dump(out,open("expA.json","w"),indent=1)
print(f"{'model':16s} acc/gen  acc/postcorr  tailG  tailA  ratio  repG  repA   sdG   sdA  sd_ratio  routeH  cf   tripwires  teacher split")
for k,v in out.items():
    g=v['ml']['generated'];a=v['ml']['accepted'];dg=v['diag']
    ng=sum(x['num_generated'] for x in dg); na=sum(x['num_accepted'] for x in dg); nc=sum(x['input_candidates'] for x in dg)
    tg=np.mean(g['tail_mass']);ta=np.mean(a['tail_mass'])
    trip=sum(x['repetition_accepted']>0.40 or x['final_route_entropy']<1.0 or x['accept_rate']<0.05 for x in dg)
    print(f"{k:16s} {na/ng:.3f}   {na/nc:.3f}        {tg:.3f} {ta:.3f} {ta/tg:.3f}  {np.mean(g['repetition']):.3f} {np.mean(a['repetition']):.3f}  {np.mean(g['entropy_std']):.3f} {np.mean(a['entropy_std']):.3f} {np.mean(a['entropy_std'])/np.mean(g['entropy_std']):.3f}    {np.mean([x['final_route_entropy'] for x in dg]):.3f}  {np.mean([x['cf_catch_rate'] for x in dg]):.2f}  {trip}/8  {v['teacher']}")
    # rounds where accepted tail > generated
    print("    tailA>tailG rounds:",sum(x>y for x,y in zip(a['tail_mass'],g['tail_mass'])),"/8; repA<repG:",sum(x<y for x,y in zip(a['repetition'],g['repetition'])),"/8; sdA<sdG:",sum(x<y for x,y in zip(a['entropy_std'],g['entropy_std'])),"/8",
          "| rej t1 mean",np.mean([x['reject_t1'] for x in dg]),"t2",np.mean([x['reject_t2'] for x in dg]),"dup",np.mean([x['reject_commit_dup'] for x in dg]), "corr-rej", np.mean([x['num_generated']-x['input_candidates'] for x in dg]))
    print("    entropy mean G",np.round(g['entropy_mean'],2).tolist()," A",np.round(a['entropy_mean'],2).tolist())
