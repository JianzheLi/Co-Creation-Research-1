import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure,load_config,load_path
from src.matching import CouplingSampler,sample_training_batch,make_feedback,matching_loss_weights
from src.experiment import make_feedback as mk
base=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve(); cand=Path(r'../Results/experiment2/eb_noscale_leakyrelu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
c0=load_config(base/'config.yaml'); c1=load_config(cand/'config.yaml'); d=configure(c0); sam=CouplingSampler(base,c0,d); path=load_path(base,c0,d)
mods=[]
for r,c in [(base,c0),(cand,c1)]:
 cp=torch.load(r/'feedback.pt',map_location=d,weights_only=False); m=mk(c,d);m.load_state_dict(cp['model']);m.eval();mods.append(m)
bins=[(0,.2),(.2,.4),(.4,.55),(.55,.7),(.7,.85),(.85,.98)]
fields=['vreg','vsing','score','growth']; buckets={n:{(a,b):{f:{k:[] for k in ['target','pred','err']} for f in fields} for a,b in bins} for n in ['silu','leaky']}
g=torch.Generator(device=d);g.manual_seed(9472)
with torch.no_grad():
 for _ in range(16):
  t,tar,fac=sample_training_batch(sam,path,2048,d,generator=g); local=(t-t.floor()).squeeze(-1); inter=t.floor().squeeze(-1)
  T={'vreg':tar['target_v_regular'].norm(dim=-1),'vsing':tar['target_v_singular_scaled'].norm(dim=-1),'score':tar['target_s_scaled'].norm(dim=-1),'growth':tar['target_g'].abs().squeeze(-1)}
  for name,m in zip(['silu','leaky'],mods):
   p=m.matching_fields(t,tar['x']); P={'vreg':p[0].norm(dim=-1),'vsing':p[1].norm(dim=-1),'score':p[3].norm(dim=-1),'growth':p[2].abs().squeeze(-1)}
   E={'vreg':(p[0]-tar['target_v_regular']).norm(dim=-1),'vsing':(p[1]-tar['target_v_singular_scaled']).norm(dim=-1),'score':(p[3]-tar['target_s_scaled']).norm(dim=-1),'growth':(p[2]-tar['target_g']).abs().squeeze(-1)}
   for a,b in bins:
    mask=(inter==1)&(local>=a)&(local<b)
    for f in fields:
     for key,v in [('target',T[f]),('pred',P[f]),('err',E[f])]: buckets[name][(a,b)][f][key].append(v[mask].detach())
def get(v):
 a=torch.cat(v).double(); return {'n':int(len(a)),'rms':float(a.square().mean().sqrt().item()),'p99':float(torch.quantile(a,.99).item()),'max':float(a.max().item())}
out={}
for name in buckets:
 out[name]={}
 for ab,fs in buckets[name].items():
  out[name][f'{ab[0]:.2f}-{ab[1]:.2f}']={}
  for f,ks in fs.items():
   o={k:get(v) for k,v in ks.items()};o['relerr']=o['err']['rms']/max(o['target']['rms'],1e-12);out[name][f'{ab[0]:.2f}-{ab[1]:.2f}'][f]=o
out['weights_equal']=matching_loss_weights(c0)==matching_loss_weights(c1)
outfile=cand/'matching_timebin_fit_comparison.json';outfile.write_text(json.dumps(out,indent=2),encoding='utf-8');
for n in ['silu','leaky']:
 print(n)
 for timebin,fs in out[n].items(): print(timebin,'Score',fs['score']['relerr'],'vreg',fs['vreg']['relerr'],'vsing',fs['vsing']['relerr'],'growth',fs['growth']['relerr'],'n',fs['score']['target']['n'])
print('weights_equal',out['weights_equal'],'saved',outfile)
