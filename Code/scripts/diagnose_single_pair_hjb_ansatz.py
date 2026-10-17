"""Independent symbolic HJB checks; no training artifacts are modified.

Convention: tau = T-t, V_tau = nu*Delta(V)-|grad(V)|^2/2-V^2/(2*alpha).
The numerical example has smooth periodic terminal data, NOT Dirac endpoints.
"""
import json
from pathlib import Path

import numpy as np
import sympy as s
from scipy.integrate import solve_ivp


def symbolic_checks():
    x, alpha, nu = s.symbols("x alpha nu", positive=True)
    a, b, h = s.symbols("a b h", real=True)
    v = a + b*x + h*x*x/2
    adot = nu*h-b*b/2-a*a/(2*alpha)
    bdot = -b*h-a*b/alpha
    hdot = -h*h-(b*b+a*h)/alpha
    residual = s.expand(adot+bdot*x+hdot*x*x/2-nu*s.diff(v,x,2)
                        +s.diff(v,x)**2/2+v*v/(2*alpha))
    assert s.simplify(residual-b*h*x**3/(2*alpha)-h*h*x**4/(8*alpha)) == 0

    vcos = a+b*s.cos(x/s.sqrt(alpha))
    adot = -(a*a+b*b)/(2*alpha)
    bdot = -nu*b/alpha-a*b/alpha
    rcos = s.trigsimp(adot+bdot*s.cos(x/s.sqrt(alpha))-nu*s.diff(vcos,x,2)
                       +s.diff(vcos,x)**2/2+vcos*vcos/(2*alpha))
    assert s.simplify(rcos) == 0

    tau, c = s.symbols("tau c", positive=True)
    q = 1+c*tau
    wfr = 2*alpha/tau*(1-s.cos(x/(2*s.sqrt(alpha)))**2/q)
    rwfr = s.trigsimp(s.expand_trig(s.together(s.diff(wfr,tau)+s.diff(wfr,x)**2/2+wfr*wfr/(2*alpha))), method="fu")
    assert s.simplify(rwfr) == 0

    n, k, p0, p2, p4, eps = s.symbols("n k p0 p2 p4 eps", real=True)
    v0 = a+k*x*x/2
    v1 = p0+p2*x*x+p4*x**4
    dv0 = nu*n*k-k*k*x*x/2
    dp4 = -4*k*p4-k*k/8
    dp2 = 4*nu*(n+2)*p4-2*k*p2-a*k/2
    dp0 = 2*nu*n*p2-a*a/2
    v = v0+eps*v1
    lap = s.diff(v,x,2)+(n-1)*s.diff(v,x)/x
    rr = s.expand(dv0+eps*(dp0+dp2*x*x+dp4*x**4)-nu*lap
                  +s.diff(v,x)**2/2+eps*v*v/2)
    assert s.simplify(rr.coeff(eps,0)) == 0
    assert s.simplify(rr.coeff(eps,1)) == 0

    # Fully explicit first reaction correction about the singular Brownian core.
    beta = s.symbols("beta", real=True)
    core_a = beta+n*nu*s.log(tau)
    core_p4 = -1/(24*tau)
    core_p2 = -core_a/4+nu*(n-4)/24
    core_p0 = tau*(-core_a**2/2+n*nu*core_a/2-nu**2*n*(5*n+4)/12)
    assert s.simplify(s.diff(core_p4,tau)+4*core_p4/tau+1/(8*tau**2)) == 0
    assert s.simplify(s.diff(core_p2,tau)-4*nu*(n+2)*core_p4
                      +2*core_p2/tau+core_a/(2*tau)) == 0
    assert s.simplify(s.diff(core_p0,tau)-2*nu*n*core_p2+core_a**2/2) == 0

    # Treat order-one mass change exactly, instead of expanding all growth.
    growth_background = 2*c/q
    curvature = 1/(tau*q)
    amplitude = (beta+n*nu*(s.log(tau)+c*tau))/q**2
    w = amplitude+curvature*x*x/2
    mass_core = alpha*growth_background+w
    radial_lap = s.diff(mass_core,x,2)+(n-1)*s.diff(mass_core,x)/x
    mass_res = (s.diff(mass_core,tau)-nu*radial_lap
                +s.diff(mass_core,x)**2/2+mass_core**2/(2*alpha))
    assert s.simplify(mass_res-w*w/(2*alpha)) == 0
    mass_p4 = -curvature/24
    assert s.simplify(s.diff(mass_p4,tau)+(4*curvature+growth_background)*mass_p4
                      +curvature**2/8) == 0

    # Exact first integral for the stationary one-dimensional equation.
    value,constant = s.symbols("value constant", real=True)
    slope_squared = constant*s.exp(value/nu)-(value**2+2*nu*value+2*nu**2)/alpha
    assert s.simplify(nu*s.diff(slope_squared,value)-slope_squared-value**2/alpha) == 0

    # Exact Gaussian-restricted Hamiltonian and centered quadratic costates (1D).
    m, variance, pm, px, ps, noise2 = s.symbols("m variance pm px ps noise2", positive=True)
    ham = (noise2*ps-m*pm**2/(2*alpha)
           -(alpha+variance)*px**2/(2*m*alpha)
           -variance*(2*alpha+variance)*ps**2/(m*alpha))
    substitution = {pm:a,px:m*b,ps:m*h/2}
    mdot = s.diff(ham,pm).subs(substitution)
    xdot = s.diff(ham,px).subs(substitution)
    sdot = s.diff(ham,ps).subs(substitution)
    adot = -s.diff(ham,m).subs(substitution)
    bdot = -mdot*b/m
    hdot = -2*s.diff(ham,variance).subs(substitution)/m-mdot*h/m
    assert s.simplify(mdot+m*a/alpha) == 0
    assert s.simplify(xdot+(1+variance/alpha)*b) == 0
    assert s.simplify(sdot-noise2+2*variance*h+variance**2*h/alpha) == 0
    assert s.simplify(adot-a*a/(2*alpha)+(1+variance/alpha)*b*b/2
                      +variance*(2*alpha+variance)*h*h/(4*alpha)) == 0
    assert s.simplify(bdot-a*b/alpha) == 0
    assert s.simplify(hdot-(a*h+b*b)/alpha-(1+variance/alpha)*h*h) == 0

    # Leading-small-diffusion covariance Euler-Lagrange equation.
    z,zd,zdd,gamma,speed2 = s.symbols("z zd zdd gamma speed2", real=True)
    lag = m*((zd-1)**2/(8*z)-z*speed2/(2*alpha))
    momentum=s.diff(lag,zd)
    el=(s.diff(momentum,m)*m*gamma+s.diff(momentum,z)*zd
        +s.diff(momentum,zd)*zdd-s.diff(lag,z))*8*z*z/m
    expected=2*z*zdd-zd**2+1+2*gamma*z*(zd-1)+4*z*z*speed2/alpha
    assert s.simplify(el-expected) == 0

    # Radial Gaussian fourth Hermite polynomial: exact chi-square moments.
    z = s.symbols("z")
    hermite = z*z-2*(n+2)*z+n*(n+2)
    def chi_expect(poly):
        return s.simplify(sum(coef*2**power*s.rf(n/2,power)
                             for (power,),coef in s.Poly(poly,z).terms()))
    assert chi_expect(hermite) == 0
    assert chi_expect(z*hermite) == 0
    assert s.simplify(chi_expect(hermite**2)-8*n*(n+2)) == 0
    return {
        "quadratic_residual": str(residual),
        "cosine_exact_residual": str(s.simplify(rcos)),
        "zero_diffusion_wfr_interior_residual": str(s.simplify(rwfr)),
        "weak_reaction_order_0_and_1_residual": "0",
        "explicit_brownian_core_first_reaction_correction": "all three coefficient equations verified",
        "mass_resummed_core_residual": "w^2/(2*alpha)",
        "mass_resummed_quartic_coefficient": "-1/(24*tau*(1+c*tau)) verified",
        "stationary_one_dimensional_first_integral": "verified",
        "gaussian_restricted_hamiltonian_canonical_equations": "verified in one dimension; diagonal dimensions add",
        "small_diffusion_gaussian_covariance_equation": str(expected),
        "radial_fourth_hermite_squared_norm_unit_variance": "8*n*(n+2)",
    }


def smooth_periodic_benchmark():
    rows = []
    for alpha in (2.,10.):
        # V(0,x)=1+0.5*cos(x/sqrt(alpha)); both modes remain regular.
        a_init,b_init = 1.,.5
        def expansion_rhs(t,y):
            a,b,da,db=y
            return [-(a*a+b*b)/(2*alpha),-a*b/alpha,
                    -(a*da+b*db)/alpha,-b/(2*alpha)-(a*db+b*da)/alpha]
        expansion=solve_ivp(expansion_rhs,(0.,1.),[a_init,b_init,0.,0.],
                            rtol=1e-12,atol=1e-14)
        assert expansion.success
        a0,b0,a1,b1=expansion.y[:,-1]
        for sigma in (.1,.25,.5,1.):
            def exact_rhs(t,y):
                a,b=y
                return [-(a*a+b*b)/(2*alpha),-sigma*sigma*b/(2*alpha)-a*b/alpha]
            exact=solve_ivp(exact_rhs,(0.,1.),[a_init,b_init],rtol=1e-12,atol=1e-14)
            assert exact.success
            ae,be=exact.y[:,-1]
            err0=abs(a0-ae)+abs(b0-be)
            err1=abs(a0+sigma*sigma*a1-ae)+abs(b0+sigma*sigma*b1-be)
            rows.append(dict(alpha=alpha,sigma=sigma,
                             zero_diffusion_value_sup_error=err0,
                             first_diffusion_correction_value_sup_error=err1,
                             corrected_error_over_sigma4=err1/sigma**4))
    return rows


if __name__ == "__main__":
    result=dict(convention="tau=T-t; V_tau=nu*Delta(V)-|grad(V)|^2/2-V^2/(2alpha)",
                scope="Symbolic identities and a smooth periodic Cauchy problem only; no hard-endpoint claim.",
                symbolic=symbolic_checks(),benchmark=smooth_periodic_benchmark())
    output=Path(__file__).resolve().parents[1]/"../Results"/"theory_hjb_ansatz"
    output.mkdir(parents=True,exist_ok=True)
    (output/"checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
