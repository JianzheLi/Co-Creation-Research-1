import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure,load_config,snapshots_for
from src.experiment import make_feedback
runs={'silu':Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve(),'leaky':Path(r'../Results/experiment2/eb_noscale_leakyrelu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve(),'scaled':Path(r'../Results/experiment2/eb_noscale_silu_depthscaled_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()}
cfg=load_config(runs['silu']/'config.yaml');d=configure(cfg);truth=snapshots_for(cfg,full=True);idx=torch.linspace(0,len(truth[0].x)-1,512,device=d).long();x0=torch.as_tensor(truth[0].x,dtype=torch.float32,device=d)[idx].clone();models={}
for name,run in runs.items():
 c=load_config(run/'config.yaml');cp=torch.load(run/'feedback.pt',map_location=d,weights_only=False);m=make_feedback(c,d);m.load_state_dict(cp['model']);m.eval();models[name]=m
# Build common state at the t=1 boundary from the original model's SDE.
x=x0.clone();gen=torch.Generator(device=d);gen.manual_seed(342);h=1/200;sig=.5
with torch.no_grad():
 for j in range(200):
  s=j*h;sn=(j+1)*h;u=math.sin(.5*math.pi*s)**2;un=math.sin(.5*math.pi*sn)**2;dt=un-u;dtds=.5*math.pi*math.sin(math.pi*s);t=torch.full((len(x),1),u,dtype=x.dtype,device=d);vr,vs,g,ss=models['silu'].matching_fields(t,x);x=x+(dtds*vr+math.pi*sig*vs+.5*math.pi*sig*ss)*h+sig*torch.randn(x.shape,dtype=x.dtype,device=d,generator=gen)*math.sqrt(dt)
xs=x
vals=[.90,.95,.98,.99,.995,.999,.9999,1.0001,1.001,.005+1,.01+1,.03+1,.05+1,.1+1,.2+1,.3+1,.4+1,.5+1,.6+1,.8+1,.9+1,.99+1]
out={}
with torch.no_grad():
 for name,m in models.items():
  rr=[]
  for tt in vals:
   t=torch.full((len(xs),1),tt,dtype=xs.dtype,device=d);vr,vs,g,ss=m.matching_fields(t,xs)
   def q(v):
    a=v.norm(dim=-1) if v.ndim==2 and v.shape[-1]>1 else v.abs().reshape(-1)
    return {'median':float(torch.quantile(a,.5).item()),'p99':float(torch.quantile(a,.99).item()),'max':float(a.max().item())}
   rr.append({'t':tt,'vreg':q(vr),'vsing':q(vs),'score':q(ss),'growth':q(g)})
  out[name]=rr
p=runs['scaled']/'evaluation_all_initial_points'/'time_profile_at_interval_boundary.json';p.write_text(json.dumps(out,indent=2),encoding='utf-8')
for name,rr in out.items(): print(name,[(r['t'],round(r['score']['max'],2),round(r['vreg']['max'],2),round(r['growth']['max'],2)) for r in rr[:15]])
print('saved',p)
