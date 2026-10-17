import itertools
import numpy as np
import pytest
import torch
from src.pair_path import PairPath, controls
from src.pair_cost import CostOracle, logmoment_bound_correction
from src.primal_coupling_gpu import perspective_value_gradient, solve_primal_gpu


@pytest.mark.parametrize('mode', ['quadratic_minimal', 'uniform_growth'])
def test_gaussian_moment_against_independent_quadrature(mode):
    torch.manual_seed(9)
    path = PairPath(dimension=2, width=8, layers=2, parameterization='axial_distance',
                    cost_objective='logmoment_upper_bound', control_mode=mode).double()
    with torch.no_grad():
        for net in (path.mu_net, path.var_net, path.mass_net):
            net.last.weight.normal_(0, .1)
    z, w = np.polynomial.hermite.hermgauss(4)
    indices = list(itertools.product(range(4), repeat=2))
    noise = torch.tensor([[z[i], z[j]] for i,j in indices])*np.sqrt(2)
    weights = torch.tensor([w[i]*w[j]/np.pi for i,j in indices])
    for t in (.01, .4, .99):
        delta = torch.tensor([[.7, -1.2]], dtype=torch.float64)
        x0 = torch.tensor([[4., -3.]], dtype=torch.float64)
        s = path(torch.tensor([[t]], dtype=torch.float64), delta, torch.tensor([[.8]], dtype=torch.float64), x0=x0)
        e = s['axis'][0]
        cov = s['var'][0,1]*torch.eye(2,dtype=torch.float64)+(s['var'][0,0]-s['var'][0,1])*torch.outer(e,e)
        y = noise@torch.linalg.cholesky(cov).T
        x = s['xi']+y
        _, g = controls(s, x, mode=mode)
        logphi = -.5*(torch.linalg.slogdet(cov)[1]+(y*torch.linalg.solve(cov,y.T).T).sum(-1))
        logq = -.5*(2*np.log(path.sigma**2*t)+(x-x0).square().sum(-1)/(path.sigma**2*t))
        integrand = .5*path.alpha*(g[:,0]+path.sigma**2/path.alpha*(logphi-logq)).square()
        expected = (weights*integrand).sum()+2*path.sigma**4/(path.alpha*np.e**2)
        actual = logmoment_bound_correction(s,path.sigma,path.alpha,mode).squeeze()
        torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)


def test_receiving_kl_gradient():
    p = torch.tensor([[.2,.3],[.1,.4]],dtype=torch.float64,requires_grad=True)
    q = torch.tensor([[.3,.2],[.2,.5]],dtype=torch.float64,requires_grad=True)
    r=q/p
    ref=torch.tensor([[.21,.29],[.21,.29]],dtype=torch.float64)
    f,gp,gq=perspective_value_gradient(p,q,p.log(),r.square(),2*r,.0625,log_reference=ref.log())
    expected=torch.autograd.grad(f,(p,q))
    torch.testing.assert_close(gp,expected[0])
    torch.testing.assert_close(gq,expected[1])


def test_bound_quadrature_and_ratio_derivative():
    path=PairPath(dimension=2,width=8,layers=2,parameterization='axial_distance',cost_objective='logmoment_upper_bound').double()
    delta=torch.tensor([[.7,-1.2],[0.,0.]],dtype=torch.float64)
    r=torch.tensor([[.2],[9.8]],dtype=torch.float64)
    values=[]
    for n in (32,64,128,256):
        oracle=CostOracle(path,nodes=n)
        assert (oracle.t<1).all()
        values.append(oracle.tensor_query(delta,r))
    for a,b in zip(values[-2],values[-1]):
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    h=1e-5
    slope=(oracle.tensor_query(delta,r+h,0)-oracle.tensor_query(delta,r-h,0))/(2*h)
    torch.testing.assert_close(slope,values[-1][1],rtol=1e-6,atol=1e-6)


def test_receiving_solver_against_independent_constrained_optimizer():
    from scipy.optimize import minimize, LinearConstraint
    a=np.array([.4,.6]); b=np.array([.55,.65])
    offset=np.array([[.1,.7],[.4,.2]])
    ref=a[:,None]*b[None,:]/b.sum()
    class Oracle:
        def value(self,r):
            return torch.as_tensor(offset,dtype=r.dtype,device=r.device)+2*(r-1).square()
        def __call__(self,r):
            return self.value(r),4*(r-1)
    actual=solve_primal_gpu(a,b,Oracle(),epsilon=.2,device='cpu',entropy_side='receiver',stationarity_tolerance=1e-6)
    assert actual.status=='converged'
    def objective(z):
        p,q=z[:4].reshape(2,2),z[4:].reshape(2,2)
        return np.sum(p*(offset+2*(q/p-1)**2)+.2*(q*np.log(q/ref)-q+ref))
    A=np.zeros((4,8)); A[0,:2]=1; A[1,2:4]=1; A[2,[4,6]]=1; A[3,[5,7]]=1
    cone1=np.concatenate((-.1*np.eye(4),np.eye(4)),axis=1)
    cone2=np.concatenate((10*np.eye(4),-np.eye(4)),axis=1)
    expected=minimize(objective,np.r_[ref.ravel(),(ref*b.sum()).ravel()],method='SLSQP',
        bounds=[(1e-10,None)]*8,constraints=[LinearConstraint(A,np.r_[a,b],np.r_[a,b]),
        LinearConstraint(cone1,0,np.inf),LinearConstraint(cone2,0,np.inf)],options={'ftol':1e-12,'maxiter':1000})
    assert expected.success
    assert abs(objective(np.r_[actual.P.ravel(),actual.Q.ravel()])-expected.fun)<1e-8


def test_cannot_load_old_path_as_bound_model(tmp_path):
    from src.runtime import load_config, make_path, load_path
    config=load_config('configs/gene_logmoment_bound.yaml')
    path=make_path(config,'cpu')
    torch.save({'model':path.state_dict(),'config':{'pair':{}}},tmp_path/'pair_path.pt')
    with pytest.raises(ValueError,match='cost objective'):
        load_path(tmp_path,config,'cpu')
