import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure, load_config, load_path
from src.matching import CouplingSampler, sample_training_batch, make_feedback
from src.experiment import make_feedback as mk_feedback

runs={
 'silu':Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve(),
 'leakyrelu':Path(r'../Results/experiment2/eb_noscale_leakyrelu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()}
cfg0=load_config(runs['silu']/'config.yaml'); dev=configure(cfg0); sampler=CouplingSampler(runs['silu'],cfg0,dev); path=load_path(runs['silu'],cfg0,dev)
truth=__import__('src.runtime',fromlist=['snapshots_for']).snapshots_for(cfg0,full=True)
idx=torch.linspace(0,len(truth[0].x)-1,512,device=dev).long(); x0=torch.as_tensor(truth[0].x,dtype=torch.float32,device=dev)[idx]
models={}
for name,run in runs.items():
 cfg=load_config(run/'config.yaml'); cp=torch.load(run/'feedback.pt',map_location=dev,weights_only=False); m=mk_feedback(cfg,dev); m.load_state_dict(cp['model']); m.eval(); models[name]=m
# matched-distribution late-interval inputs
G=torch.Generator(device=dev); G.manual_seed(9917)
late=[]
with torch.no_grad():
 for _ in range(8):
  t,tar,_=sample_training_batch(sampler,path,2048,dev,generator=G)
  local=t-t.floor(); mask=(t.floor().squeeze(-1)==1)&(local.squeeze(-1)>.75)&(local.squeeze(-1)<.98)
  if mask.any(): late.append((t[mask],tar['x'][mask]))
 train_t=torch.cat([v[0] for v in late]); train_x=torch.cat([v[1] for v in late])
def hidden_layers(net,t,x):
 if net.network_variant=='fourier':
  freq=t.new_tensor((1.,2.,4.,8.)); ph=(2*math.pi/net.time_scale)*t*freq; ti=torch.cat((t,ph.sin(),ph.cos()),dim=-1)
 else: ti=t
 h=torch.cat((ti,x),dim=-1); norms=[]
 for k in range((len(net.net)-1)//2):
  z=net.net[2*k](h)
  if k>0 and net.network_variant=='residual': z=z+h
  elif k>0 and net.network_variant in {'gated','gated_residual'}:
   gate=torch.sigmoid(net.gates[k-1](h))
   z=gate*z+(1-gate)*h if net.network_variant=='gated' else h+gate*z
  h=net.net[2*k+1](z); norms.append(h.norm(dim=-1))
 y=net.net[-1](h)
 return norms,y
out={}
for name,model in models.items():
 # capture a real SDE input at the first anomalous score peak during interval 1
 x=x0.clone(); gen=torch.Generator(device=dev); gen.manual_seed(342); found=None; steps=200; ds=1/steps; sigma=cfg0['model_sigma']
 with torch.no_grad():
  for interval in range(2):
   for j in range(steps):
    s=j*ds; sn=(j+1)*ds; local=math.sin(.5*math.pi*s)**2; localn=math.sin(.5*math.pi*sn)**2; delta=localn-local; dtds=.5*math.pi*math.sin(math.pi*s)
    t=torch.full((len(x),1),interval+local,dtype=x.dtype,device=dev); vr,vs,g,ss=model.matching_fields(t,x)
    score_norm=ss.norm(dim=-1); mx,ix=score_norm.max(0)
    if interval==1 and found is None and mx.item()>100:
     found={'interval':interval,'step':j+1,'time':interval+local,'particle':int(ix.item()),'score_scaled_norm':float(mx.item()),'x_norm':float(x[ix].norm().item()),'x_abs_max':float(x[ix].abs().max().item()),'x':x[ix].detach().clone()}
    x=x+(.5*math.pi*math.sin(math.pi*s)*vr+math.pi*sigma*vs+.5*math.pi*sigma*ss)*ds+sigma*torch.randn(x.shape,dtype=x.dtype,device=dev,generator=gen)*math.sqrt(delta)
    if not torch.isfinite(x).all(): break
   if not torch.isfinite(x).all(): break
 if found is None: raise RuntimeError(f'{name}: no score>100 found before divergence')
 et=torch.full((1,1),found['time'],dtype=torch.float32,device=dev)
 ood_norms,ood_score=hidden_layers(model.score,et,found['x'].reshape(1,-1))
 tr_norms,tr_score=hidden_layers(model.score,train_t,train_x)
 def quant(v): return {'p50':float(torch.quantile(v,.5).item()),'p99':float(torch.quantile(v,.99).item()),'max':float(v.max().item())}
 found['x']=found['x'].cpu().tolist(); found['hidden_norms_actual_outlier']=[float(v.item()) for v in [n[0] for n in ood_norms]]; found['score_out_norm_actual_outlier']=float(ood_score.norm().item())
 found['hidden_norms_training_late']=[quant(n) for n in tr_norms]; found['score_norm_training_late']=quant(tr_score.norm(dim=-1))
 out[name]=found
path=runs['leakyrelu']/'evaluation_all_initial_points'/'score_hidden_activation_diagnostic.json'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2),encoding='utf-8')
for name,r in out.items():
 print(name,json.dumps({k:v for k,v in r.items() if k!='x'}))
print('saved',path)
