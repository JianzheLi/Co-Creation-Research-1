import json
from pathlib import Path
for prefix in ['gene_distance_hermite_primal','mouse_distance_hermite_primal','emt_distance_hermite_primal','gene_wfr_bridge','mouse_wfr_bridge','emt_wfr_bridge']:
 p=Path('../Results')/f'{prefix}_alpha2_sigma1_seed42'/'status.json'
 x=json.loads(p.read_text())
 print(p.parent.name,{k:v.get('status') for k,v in x.items() if isinstance(v,dict)})
