import json
from pathlib import Path
for dataset,run in [('gene','gene_distance_hermite_primal_alpha2_seed42'),('mouse','mouse_distance_hermite_primal_seed42'),('emt','emt_distance_hermite_primal_seed42')]:
 p=Path('../Results')/run/'conditional_path_comparison.json'
 if not p.exists(): print(dataset,'missing'); continue
 d=json.loads(p.read_text()); print(dataset,'alpha',d['alpha'],'sigma',d['sigma'])
 for k in ['mean_shift_rms','mass_relative_rms','variance_relative_rms']:
  vals=[i[k] for i in d['intervals']]; print(k,sum(vals)/len(vals))
