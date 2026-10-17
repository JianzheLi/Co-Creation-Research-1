import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure, load_config, snapshots_for
from src.experiment import make_feedback
runs={'silu':Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve(),'leaky':Path(r'../Results/experiment2/eb_noscale_leakyrelu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()}
cfg=load_config(runs['silu']/'config.yaml'); dev=configure(cfg); truth=snapshots_for(cfg,full=True); idx=torch.linspace(0,len(truth[0].x)-1,512,device=dev).long(); x0=torch.as_tensor(truth[0].x,dtype=torch.float32,device=dev)[idx]
def gethidden(net,t,x):
 ti=t; h=torch.cat((ti,x),-1); vals=[]
 for k in range((len(net.net)-1)//2):
  z=net.net[2*k](h)
  if k>0 and net.network_variant=='residual':z=z+h
  h=net.net[2*k+1](z); vals.append(float(h.norm(dim=-1)[0].item()))
 return vals, float(net.net[-1](h).norm(dim=-1)[0].item())
allout={}
for name,run in runs.items():
 c=load_config(run/'config.yaml'); cp=torch.load(run/'feedback.pt',map_location=dev,weights_only=False); m=make_feedback(c,dev); m.load_state_dict(cp['model']); m.eval()
 x=x0.clone(); lw=torch.full((len(x),),-math.log(len(x)),dtype=torch.float64,device=dev); gen=torch.Generator(device=dev); gen.manual_seed(342); rows=[]; anomaly=None; failed=None; ds=1/200; sig=c['model_sigma']
 with torch.no_grad():
  for interval in range(4):
   for j in range(200):
    s=j*ds; sn=(j+1)*ds; u=math.sin(.5*math.pi*s)**2; un=math.sin(.5*math.pi*sn)**2; delta=un-u; dtds=.5*math.pi*math.sin(math.pi*s); t=torch.full((len(x),1),interval+u,dtype=x.dtype,device=dev)
    vr,vs,g,ss=m.matching_fields(t,x); norms={'vreg':vr.norm(dim=-1),'vsing':vs.norm(dim=-1),'score':ss.norm(dim=-1),'growth':g.abs().squeeze(-1)}
    vals={k:float(v.max().item()) for k,v in norms.items()}; x_abs=float(x.abs().max().item()); xnorm=float(x.norm(dim=-1).max().item()); lwmax=float(lw.abs().max().item())
    if (max(vals['score'],vals['vreg'],vals['vsing'])>50 or vals['growth']>2 or x_abs>100 or lwmax>8) and (j%5==0 or anomaly is None):
     rows.append({'interval':interval,'step':j,'time':interval+u,'x_abs':x_abs,'xnorm_max':xnorm,'lw_abs':lwmax,**vals})
    if anomaly is None and (vals['score']>50 or vals['vreg']>50 or vals['growth']>2 or x_abs>100 or lwmax>8):
     key=max(vals,key=vals.get); v=norms[key]; ix=int(v.argmax().item()); net=m.score if key=='score' else m.growth if key=='growth' else m.flow_regular if key=='vreg' and m.independent_velocity_networks else m.flow_singular
     input_t=t[ix:ix+1]; input_x=x[ix:ix+1].clone(); hidden,out=gethidden(net,input_t,input_x)
     anomaly={'trigger_field':key,'trigger_value':vals[key],'interval':interval,'step':j,'time':interval+u,'particle':ix,'x_abs':float(input_x.abs().max().item()),'xnorm':float(input_x.norm().item()),'trigger_hidden_norms':hidden,'trigger_network_output_norm':out,'field_values':vals}
    drift=.5*math.pi*math.sin(math.pi*s)*vr+math.pi*sig*vs+.5*math.pi*sig*ss
    x=x+drift*ds+sig*torch.randn(x.shape,dtype=x.dtype,device=dev,generator=gen)*math.sqrt(delta); lw=lw+(dtds*g.squeeze(-1).double())*ds
    if not torch.isfinite(x).all() or not torch.isfinite(lw).all():
     failed={'interval':interval,'step':j,'time':interval+un,'x_finite':bool(torch.isfinite(x).all()),'logw_finite':bool(torch.isfinite(lw).all()),'x_abs':float(x.abs().max().item()),'logw_abs':float(lw.abs().max().item()),'field_values':vals}; break
   if failed:break
 allout[name]={'anomaly':anomaly,'failure':failed,'high_field_rows':rows}
 print(name,json.dumps({'anomaly':anomaly,'failure':failed},indent=2),flush=True)
outfile=runs['leaky']/'evaluation_all_initial_points'/'per_field_failure_trace.json'; outfile.parent.mkdir(parents=True,exist_ok=True); outfile.write_text(json.dumps(allout,indent=2),encoding='utf-8'); print('saved',outfile)
