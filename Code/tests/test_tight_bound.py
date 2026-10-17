import numpy as np
import pytest
import torch
from scipy.integrate import quad
from src.pair_path import PairPath,controls,spectral_apply
from src.pair_cost import finite_moments,finite_pair_integrand,CostOracle
from src.tight_bound import normal_positive_quadratic_second,positive_log_moment,log_ratio_moments
from src.matching import make_targets
from src.primal_coupling_gpu import solve_primal_gpu


def path_and_state(dimension=2):
    torch.manual_seed(17)
    p=PairPath(dimension=dimension,width=8,layers=2,parameterization='axial_distance',cost_objective='tight_logmoment_upper_bound').double()
    with torch.no_grad():
        for net in (p.mu_net,p.var_net,p.mass_net):
            net.last.weight.normal_(0,.08)
    t=torch.tensor([[.1],[.5],[.9]],dtype=torch.float64)
    delta=torch.randn(3,dimension,dtype=torch.float64)
    s=p(t,delta,torch.tensor([[-.3],[.7],[1.2]],dtype=torch.float64))
    return p,s


@pytest.mark.parametrize('abc',[(1.,2.,-.4),(-1.,.2,2.),(-1.,.3,-2.),(.01,4.,1.),(0.,2.,1.),(0.,-2.,1.),(0.,0.,-1.),(0.,0.,1.)])
def test_positive_normal_quadratic(abc):
    a,b,c=abc
    expected=quad(lambda x:max(a*x*x+b*x+c,0)**2*np.exp(-x*x/2)/np.sqrt(2*np.pi),-12,12,epsabs=1e-10,points=np.roots([a,b,c]).real.tolist() if a and b*b-4*a*c>0 else None)[0]
    actual=normal_positive_quadratic_second(*[torch.tensor(v,dtype=torch.float64,requires_grad=True) for v in abc])
    assert abs(actual.item()-expected)<1e-7


@pytest.mark.parametrize('dimension',[2,10])
def test_kkt_pde_matching_and_endpoints(dimension):
    p,s=path_and_state(dimension)
    noise=torch.randn_like(s['delta'])
    target=make_targets(s,noise)
    x=target['x'].detach().requires_grad_()
    u,g=controls(s,x)
    y=x-s['xi']
    score=-spectral_apply(s,1/s['var'],y)
    logphi=-.5*(y*spectral_apply(s,1/s['var'],y)).sum(-1,keepdim=True)-.5*s['var'].log().sum(-1,keepdim=True)
    logq=-.5*x.square().sum(-1,keepdim=True)/(p.sigma**2*s['t'])-.5*dimension*(p.sigma**2*s['t']).log()
    potential=2*p.alpha*g+p.sigma**2*(logphi-logq)
    gradient=torch.autograd.grad(potential.sum(),x,retain_graph=True)[0]
    torch.testing.assert_close(u,gradient,atol=1e-10,rtol=1e-10)
    divu=sum(torch.autograd.grad(u[:,i].sum(),x,retain_graph=True)[0][:,i:i+1] for i in range(dimension))
    rhs=-divu-(u*score).sum(-1,keepdim=True)+p.sigma**2/2*(score.square().sum(-1,keepdim=True)-(1/s['var']).sum(-1,keepdim=True))+g
    invy=spectral_apply(s,1/s['var'],y)
    lhs=s['gamma']+(s['dotxi']*invy).sum(-1,keepdim=True)+.5*((invy*spectral_apply(s,s['dotvar'],invy)).sum(-1,keepdim=True)-(s['dotvar']/s['var']).sum(-1,keepdim=True))
    torch.testing.assert_close(lhs,rhs,atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(target['target_g'],g)
    torch.testing.assert_close(target['target_v_scaled'],target['sqrt_wf']*(u-p.sigma**2/2*score))
    for t in (1e-8,1-1e-8):
        endpoint=p(torch.full_like(s['t'],t),s['delta'],s['log_r'])
        moments=finite_moments(endpoint)
        assert all(torch.isfinite(v).all() for v in moments.values())


@pytest.mark.parametrize('dimension',[2,10])
def test_positive_moment_and_quadrature_convergence(dimension):
    p,s=path_and_state(dimension)
    full=log_ratio_moments(s,p.sigma)[1]
    low=positive_log_moment(s,p.sigma,32); high=positive_log_moment(s,p.sigma,128)
    assert (low>=0).all() and (low<=full+1e-6).all()
    torch.testing.assert_close(low,high,atol=1e-3,rtol=1e-4)
    # Compare the final cost, slope and curvature, not only raw moment values.
    p.logmoment_nodes=32
    a=CostOracle(p,nodes=32,condition_chunk=4).tensor_query(s['delta'],s['log_r'].exp())
    p.logmoment_nodes=128
    b=CostOracle(p,nodes=64,condition_chunk=4).tensor_query(s['delta'],s['log_r'].exp())
    for v,w in zip(a,b):
        torch.testing.assert_close(v,w,atol=1e-4,rtol=1e-4)


def test_conditional_entropy_solver_against_scipy():
    from scipy.optimize import minimize,LinearConstraint
    a=np.array([.4,.6]); b=np.array([.55,.65]); offset=np.array([[.1,.7],[.4,.2]])
    class Oracle:
        def value(self,r): return torch.as_tensor(offset,dtype=r.dtype)+2*(r-1).square()
        def __call__(self,r): return self.value(r),4*(r-1)
    result=solve_primal_gpu(a,b,Oracle(),epsilon=.2,device='cpu',entropy_side='receiver_conditional',stationarity_tolerance=1e-6)
    assert result.status=='converged'
    def objective(z):
        p,q=z[:4].reshape(2,2),z[4:].reshape(2,2)
        return np.sum(p*(offset+2*(q/p-1)**2)+.2*q*np.log(q/(q.sum(1)[:,None]*b[None,:]/b.sum())))
    A=np.zeros((4,8)); A[0,:2]=1; A[1,2:4]=1; A[2,[4,6]]=1; A[3,[5,7]]=1
    ref=a[:,None]*b[None,:]/b.sum()
    expected=minimize(objective,np.r_[ref.ravel(),(ref*b.sum()).ravel()],method='SLSQP',bounds=[(1e-10,None)]*8,
        constraints=[LinearConstraint(A,np.r_[a,b],np.r_[a,b]),
        LinearConstraint(np.c_[-.1*np.eye(4),np.eye(4)],0,np.inf),LinearConstraint(np.c_[10*np.eye(4),-np.eye(4)],0,np.inf)],
        options={'ftol':1e-12,'maxiter':1000})
    assert expected.success
    assert abs(objective(np.r_[result.P.ravel(),result.Q.ravel()])-expected.fun)<1e-8


def test_finite_moments_independently_and_bound_dominance():
    import itertools
    p,s=path_and_state(2)
    z,w=np.polynomial.hermite.hermgauss(5)
    indices=list(itertools.product(range(5),repeat=2))
    noise=torch.tensor([[z[i],z[j]] for i,j in indices])*np.sqrt(2)
    weights=torch.tensor([w[i]*w[j]/np.pi for i,j in indices])
    for j in range(3):
        sj={k:v[j:j+1] if torch.is_tensor(v) else v for k,v in s.items()}
        x=sj['xi']+spectral_apply(sj,sj['var'].sqrt(),noise)
        u,g=controls(sj,x)
        x1=sj['delta']
        F=(x-x1).square().sum(-1,keepdim=True)/(2*(1-sj['t']))
        residual=u+(x-x1)/(1-sj['t'])
        actual=finite_moments(sj)
        for key,values in [('E_res',residual.square().sum(-1)),('E_g2',g[:,0].square()),('E_gF',(g*F)[:,0])]:
            torch.testing.assert_close(actual[key].squeeze(),(weights*values).sum(),atol=1e-10,rtol=1e-10)
    old=dict(s,cost_objective='logmoment_upper_bound')
    assert (finite_pair_integrand(s)<=finite_pair_integrand(old)+1e-8).all()


def test_positive_part_mixture_bound():
    x=np.linspace(-16,16,160001); dx=x[1]-x[0]
    q=np.exp(-x*x/2)/np.sqrt(2*np.pi)
    densities=np.stack([np.exp(-(x-mu)**2/(2*sd**2))/(sd*np.sqrt(2*np.pi))
                        for mu,sd in [(-1.,.5),(.4,1.2),(2.,.8)]])
    weights=np.array([.2,.5,.3]); mixture=weights@densities
    ell=np.log(densities/q)
    lhs=np.sum(mixture*np.log(mixture/q)**2)*dx
    new=np.sum(weights[:,None]*densities*np.maximum(ell,0)**2)*dx+4/np.e**2
    old=np.sum(weights[:,None]*densities*ell**2)*dx+4/np.e**2
    assert lhs<=new<=old
