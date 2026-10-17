
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code: src/
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.runtime import load_config
for f in sys.argv[1:]:
 c=load_config(f)
 print(f, c['dataset'], 'alpha=',c['growth_penalty'],'sigma=',c['model_sigma'],'steps=',c['pair']['training_steps'],'coupling=',c['coupling']['mode'],c['coupling'].get('minibatch_blocks'))
