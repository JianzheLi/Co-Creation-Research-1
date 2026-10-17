import torch
from src.runtime import load_config, make_path
from src.matching import make_targets
from src.pair_path import controls, spectral_apply
from src.pair_cost import finite_moments


def test_alpha2_configs_and_matching_share_control_coefficient():
    ruot = load_config('configs/gene_distance_hermite_primal_alpha2.yaml')
    wfr = load_config('configs/gene_wfr_bridge_alpha2.yaml')
    for key in ('growth_penalty','model_sigma','matching','generation','model_times','data_path'):
        assert ruot[key] == wfr[key]
    assert ruot['growth_penalty'] == 2
    assert ruot['pair']['training_steps'] == 3000
    path = make_path(ruot, torch.device('cpu'))
    state = path(torch.tensor([[.43]],dtype=torch.float64),
                 torch.tensor([[.7,-.3]],dtype=torch.float64),
                 torch.tensor([[.5]],dtype=torch.float64))
    noise = torch.tensor([[1.2,-.8]],dtype=torch.float64)
    labels = make_targets(state,noise,path.sigma,path.alpha)
    drift,growth = controls(state,labels['x'],path.alpha)
    score = -spectral_apply(state,1/state['var'],labels['x']-state['xi'])
    torch.testing.assert_close(labels['target_g'],growth)
    torch.testing.assert_close(labels['target_v_scaled'],state['tau'].sqrt()*(drift-.5*path.sigma**2*score))
    moments = finite_moments(state,path.sigma,path.alpha)
    variance = ((state['b']*spectral_apply(state,state['var'],state['b'])).sum(-1,keepdim=True)
                +.5*state['K'].square().sum(-1,keepdim=True))/path.alpha**2
    torch.testing.assert_close(moments['E_g2'],state['gamma'].square()+variance)
