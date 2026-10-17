import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure,load_config,snapshots_for
from src.experiment import make_feedback
run=Path(r'../Results/experiment2/eb_noscale_silu_depthscaled_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve(); c=load_config(run/'config.yaml');d=configure(c);truth=snapshots_for(c,full=True);cp=torch.load(run/'feedback.pt',map_location=d,weights_only=False);m=make_feedback(c,d);m.load_state_dict(cp['model']);m.eval();idx=torch.linspace(0,len(truth[0].x)-1,512,device=d).long();x=torch.as_tensor(truth[0].x,dtype=torch.float32,device=d)[idx].clone();lw=torch.full((len(x),),-math.log(len(x)),dtype=torch.float64,device=d);gen=torch.Generator(device=d);gen.manual_seed(342);rows=[];ds=1/200;sig=c['model_sigma'];failure=None
with torch.no_grad():
 for k in range(4):
  for j in range(200):
   s=j*ds;sn=(j+1)*ds;u=math.sin(.5*math.pi*s)**2;un=math.sin(.5*math.pi*sn)**2;dt=un-u;dtds=.5*math.pi*math.sin(math.pi*s);t=torch.full((len(x),1),k+u,dtype=x.dtype,device=d)
   vr,vs,g,ss=m.matching_fields(t,x); vals={'vreg':float(vr.norm(dim=-1).max().item()),'vsing':float(vs.norm(dim=-1).max().item()),'score':float(ss.norm(dim=-1).max().item()),'growth':float(g.abs().max().item())}
   if k==0 and (j>0 and j%10==0) or max(vals.values())>50 or vals['growth']>2 or float(x.abs().max())>100:
    rows.append({'interval':k,'step':j,'time':k+u,'xabs':float(x.abs().max()),'xnorm':float(x.norm(dim=-1).max()),**vals})
   x=x+(dtds*vr+math.pi*sig*vs+.5*math.pi*sig*ss)*ds+sig*torch.randn(x.shape,dtype=x.dtype,device=d,generator=gen)*math.sqrt(dt);lw=lw+(dtds*g.squeeze(-1).double())*ds
   if not torch.isfinite(x).all() or not torch.isfinite(lw).all():
    failure={'interval':k,'step':j,'time':k+un,'x_finite':bool(torch.isfinite(x).all()),'logw_finite':bool(torch.isfinite(lw).all()),'xabs':float(x.abs().max()),'logwabs':float(lw.abs().max()),'fields':vals};break
  if failure:break
out={'failure':failure,'rows':rows}; p=run/'evaluation_all_initial_points'/'scaled_residual_failure_trace_seed342.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
