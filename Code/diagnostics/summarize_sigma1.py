import json
from pathlib import Path
import pandas as pd
root=Path('../Results')
for d in ['gene','mouse','emt']:
 a=root/f'{d}_distance_hermite_primal_alpha2_sigma1_seed42'
 b=root/f'{d}_wfr_bridge_alpha2_sigma1_seed42'
 ar=json.loads((a/'action.json').read_text()); br=json.loads((b/'action.json').read_text())
 rm=pd.read_csv(a/'metrics.csv'); wm=pd.read_csv(b/'metrics.csv')
 print(d, 'Action RUOT/WFR/diff=',ar['total_action'],br['total_action'],ar['total_action']-br['total_action'], 'SE=',ar['particle_mc_standard_error'],br['particle_mc_standard_error'])
 for k in ['W1_normalized','W2_normalized','mass_relative_error']:
  x=rm.loc[rm.time>0,k]; y=wm.loc[wm.time>0,k]
  print(k,'RUOT',x.tolist(),'WFR',y.tolist(),'mean',x.mean(),y.mean())
 p=json.loads((a/'conditional_path_comparison.json').read_text())
 print('path', {key:sum(i[key] for i in p['intervals'])/len(p['intervals']) for key in ['mean_shift_rms','mass_relative_rms','variance_relative_rms']},'cost weighted',p['weighted_pair_surrogates_on_ruot_coupling'])
