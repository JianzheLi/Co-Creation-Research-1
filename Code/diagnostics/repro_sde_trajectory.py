import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure,load_config,snapshots_for
from src.experiment import make_feedback
run=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve();c=load_config(run/'config.yaml');d=configure(c);truth=snapshots_for(c,full=True);cp=torch.load(run/'feedback.pt',map_location=d,weights_only=False);m=make_feedback(c,d);m.load_state_dict(cp['model']);m.eval();idx=torch.linspace(0,len(truth[0].x)-1,512,device=d).long();x0=torch.as_tensor(truth[0].x,dtype=torch.float32,device=d)[idx].clone();outputs=[]
for rep in range(2):
 x=x0.clone();g=torch.Generator(device=d);g.manual_seed(342);h=1/200;rec={}
 with torch.no_grad():
  for k in range(2):
   for j in range(130):
    s=j*h;sn=(j+1)*h;u=math.sin(.5*math.pi*s)**2;un=math.sin(.5*math.pi*sn)**2;dt=un-u;dtds=.5*math.pi*math.sin(math.pi*s);t=torch.full((len(x),1),k+u,dtype=x.dtype,device=d);vr,vs,growth,ss=m.matching_fields(t,x)
    if k==1 and j in [88,92,100,110,120,125]:rec[str(j)]={'x':x[239].clone(),'score':ss[239].norm().item(),'max_score':ss.norm(dim=-1).max().item()}
    x=x+(dtds*vr+math.pi*c['model_sigma']*vs+.5*math.pi*c['model_sigma']*ss)*h+c['model_sigma']*torch.randn(x.shape,dtype=x.dtype,device=d,generator=g)*math.sqrt(dt)
 outputs.append(rec)
res=[]
for j in outputs[0]: res.append({'step':j,'x_difference':float((outputs[0][j]['x']-outputs[1][j]['x']).norm().item()),'score1':outputs[0][j]['score'],'score2':outputs[1][j]['score'],'max1':outputs[0][j]['max_score'],'max2':outputs[1][j]['max_score']})
print(json.dumps(res,indent=2))
