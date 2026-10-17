import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.runtime import load_config, init_run, json_write, file_hash
source=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
config=load_config(source/'config.yaml'); config['matching']['network_variant']='residual_scaled'
config_path=Path(r'configs/experiment2_eb_noscale_silu_depthscaled_residual_w512x8_a1225_sigma05_blocks8_fm30000.yaml').resolve()
config_path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
dest=Path(r'../Results/experiment2/eb_noscale_silu_depthscaled_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
if dest.exists(): raise FileExistsError(dest)
init_run(dest,config)
for name in ('pair_path.pt','data_metadata.json','quadrature_validation.json','pair_validation.jsonl','pair_steps.jsonl','distance_cost_table.pt','distance_table_validation.json'):
 if (source/name).exists(): shutil.copy2(source/name,dest/name)
shutil.copytree(source/'couplings',dest/'couplings')
old=json.loads((source/'status.json').read_text(encoding='utf-8')); json_write(dest/'status.json',{'pair':old['pair'],'couplings':old['couplings']})
json_write(dest/'provenance.json',{'reused_pair_and_coupling_from':str(source),'pair_checkpoint_sha256':file_hash(dest/'pair_path.pt'),'coupling_manifest_sha256':file_hash(dest/'couplings'/'manifest.json'),'matching_only_retrained':True,'only_matching_network_variant_changed':['residual','residual_scaled'],'fixed_residual_branch_scale':'1/(hidden_layers-1)','matching_loss_and_weights_unchanged':True})
print(json.dumps({'config':str(config_path),'run':str(dest),'variant':config['matching']['network_variant'],'activation':config['matching']['feedback_activation'],'growth_weight':config['growth_penalty']*config['matching']['growth_loss_multiplier']},indent=2))
