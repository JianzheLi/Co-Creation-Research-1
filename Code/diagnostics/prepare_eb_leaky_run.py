import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config, init_run, json_write, file_hash
import yaml
source=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
config_path=Path(r'configs/experiment2_eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000.yaml').resolve()
config=load_config(config_path)
config['matching']['feedback_activation']='leaky_relu'
config['matching']['last_hidden_activation']='leaky_relu'
new_config=Path(r'configs/experiment2_eb_noscale_leakyrelu_residual_w512x8_a1225_sigma05_blocks8_fm30000.yaml').resolve()
new_config.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
dest=Path(r'../Results/experiment2/eb_noscale_leakyrelu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
if dest.exists(): raise FileExistsError(dest)
init_run(dest,config)
for name in ('pair_path.pt','data_metadata.json','quadrature_validation.json','pair_validation.jsonl','pair_steps.jsonl'):
    if (source/name).exists(): shutil.copy2(source/name,dest/name)
shutil.copy2(source/'distance_cost_table.pt',dest/'distance_cost_table.pt')
shutil.copy2(source/'distance_table_validation.json',dest/'distance_table_validation.json')
shutil.copytree(source/'couplings',dest/'couplings')
old=json.loads((source/'status.json').read_text(encoding='utf-8'))
status={'pair':old['pair'],'couplings':old['couplings']}
json_write(dest/'status.json',status)
json_write(dest/'provenance.json',{'reused_pair_and_coupling_from':str(source),'pair_checkpoint_sha256':file_hash(dest/'pair_path.pt'),'coupling_manifest_sha256':file_hash(dest/'couplings'/'manifest.json'),'matching_only_retrained':True,'only_matching_activation_changed':['silu','silu','residual','leaky_relu','leaky_relu'],'matching_loss_and_weights_unchanged':True})
print(json.dumps({'config':str(new_config),'run':str(dest),'pair_sha256':file_hash(dest/'pair_path.pt'),'coupling_manifest_sha256':file_hash(dest/'couplings'/'manifest.json'),'status':status},indent=2))
