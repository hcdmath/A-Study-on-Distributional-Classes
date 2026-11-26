"""Reproducible log-utility benchmark and sampled q-learning ablation.

Only simulator() and benchmark() access market coefficients. Learners receive
sampled states, actions and rewards; both ablations use identical data streams.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import gammaln, digamma
from numpy.polynomial.legendre import legvander

T=1.0
RHO=0.05
LAM=0.10
GT=np.exp(-RHO*T)
MARKET={'R':np.array([.02,.01]),'mu':np.array([.08,.04]),
        'sigma':np.array([.20,.35]),'Q':np.array([[-.8,.8],[.5,-.5]])}
ROOT=Path(__file__).resolve().parent

def gamma_t(t): return np.exp(-RHO*np.asarray(t))
def exact_A(t, lam=LAM):
    t=np.asarray(t)
    return GT+(gamma_t(t)-GT)/RHO+lam*(T-t)

def source_F(t, lam=LAM):
    t=np.asarray(t); a=exact_A(t,lam); g=gamma_t(t); alpha=1+g/lam
    r=MARKET['R']; sig=MARKET['sigma']; th=MARKET['mu']-r
    common=lam*gammaln(alpha)+(g+lam)*np.log(lam/a)
    return a[...,None]*(r+th**2/(2*sig**2))+common[...,None]+lam/2*np.log(2*np.pi*lam/(a[...,None]*sig**2))

def benchmark(lam=LAM):
    sol=solve_ivp(lambda t,b:-MARKET['Q']@b-source_F(t,lam),[T,0],np.zeros(2),
                  method='DOP853',rtol=2e-12,atol=2e-13,dense_output=True)
    if not sol.success: raise RuntimeError(sol.message)
    return sol

def basis(t): return legvander(2*np.asarray(t)/T-1,3)

def value_features(t,y,i):
    f=(T-np.asarray(t))[:,None]*basis(t)
    return np.column_stack((f*np.asarray(y)[:,None],f*(np.asarray(i)==0)[:,None],f*(np.asarray(i)==1)[:,None]))

def policy_eval(w,t,c,p,i,free_shape=False):
    f=basis(t)
    u=f@w[:4]
    mean=w[4+np.asarray(i)]
    lv=np.einsum('ij,ij->i',f,w[6:14].reshape(2,4)[i])
    alpha=(1+gamma_t(t)/LAM)*np.exp(f@w[14:18] if free_shape else 0.)
    rate=alpha*np.exp(u)
    var=np.exp(lv)
    log_gamma=alpha*np.log(rate)-gammaln(alpha)+(alpha-1)*np.log(c)-rate*c
    log_normal=-.5*(np.log(2*np.pi)+lv+(p-mean)**2/var)
    q_relative=LAM*(log_gamma+log_normal)
    grad=np.zeros((len(t),len(w)))
    grad[:,:4]=LAM*(alpha-rate*c)[:,None]*f
    for regime in [0,1]:
        mask=(i==regime)
        grad[mask,4+regime]=LAM*(p[mask]-mean[mask])/var[mask]
        grad[mask,6+4*regime:10+4*regime]=(.5*LAM*((p[mask]-mean[mask])**2/var[mask]-1))[:,None]*f[mask]
    if free_shape:
        score=alpha*(np.log(rate)+1-digamma(alpha)+np.log(c)-np.exp(u)*c)
        grad[:,14:18]=LAM*score[:,None]*f
    return q_relative,grad,alpha,rate,mean,var

def initial_parameters(free_shape):
    phi=np.zeros(12);phi[0]=.50
    w=np.zeros(18 if free_shape else 14)
    w[0]=-np.log(.80)
    w[4:6]=.75
    w[6]=w[10]=np.log(.80)
    return phi,w

class AdamAscent:
    def __init__(self,n):self.m=np.zeros(n);self.v=np.zeros(n);self.n=0
    def step(self,x,g,lr):
        self.n+=1
        norm=np.linalg.norm(g)
        if norm>10:g=g*10/norm
        self.m=.9*self.m+.1*g;self.v=.999*self.v+.001*g*g
        x+=lr*(self.m/(1-.9**self.n))/(np.sqrt(self.v/(1-.999**self.n))+1e-8)

def simulator(rng,h,batch):
    """Log-wealth splitting step, with paired opposite Brownian shocks.

    This is a positivity-preserving approximation, not an exact constant-
    absolute-consumption transition. Its local generator agrees to first order.
    Each action produces two observed transitions; the regime draw is shared.
    """
    K=round(T/h)
    t=rng.integers(0,K,size=batch)*h
    y=rng.uniform(-1,1,size=batch)
    i=rng.integers(0,2,size=batch)
    c=rng.gamma(4.,.25,size=batch)
    p=rng.normal(0.,2.,size=batch)
    z=rng.standard_normal(batch)
    P=expm(MARKET['Q']*h)
    j=(rng.uniform(size=batch)>P[i,0]).astype(int)
    sig=MARKET['sigma'][i];r=MARKET['R'][i];th=MARKET['mu'][i]-r
    drift=r+th*p-c-.5*sig**2*p*p
    ym=y+drift*h
    yplus=ym+sig*p*np.sqrt(h)*z
    yminus=ym-sig*p*np.sqrt(h)*z
    reward=gamma_t(t)*(y+np.log(c))*h
    return t,y,i,c,p,j,yplus,yminus,reward

def evaluate(phi,w,free_shape,sol):
    tt=np.linspace(0,T,101)
    t=np.repeat(tt,2); i=np.tile([0,1],len(tt))
    _,_,aq,bq,mq,vq=policy_eval(w,t,np.ones_like(t),np.zeros_like(t),i,free_shape)
    ap=1+gamma_t(t)/LAM;bp=exact_A(t)/LAM
    mp=(MARKET['mu'][i]-MARKET['R'][i])/MARKET['sigma'][i]**2
    vp=LAM/(exact_A(t)*MARKET['sigma'][i]**2)
    klg=(ap-aq)*digamma(ap)-gammaln(ap)+gammaln(aq)+aq*np.log(bp/bq)+ap*(bq/bp-1)
    kln=.5*(np.log(vq/vp)+(vp+(mp-mq)**2)/vq-1)
    ys=np.linspace(-1,1,11)
    t3=np.repeat(t,len(ys));i3=np.repeat(i,len(ys));y3=np.tile(ys,len(t))
    pred=GT*y3+value_features(t3,y3,i3)@phi
    truth=exact_A(t3)*y3+sol.sol(t3)[i3,np.arange(len(t3))]
    rrmse=np.sqrt(np.mean((pred-truth)**2)/np.mean(truth**2))
    return {'kl':float(np.mean(klg+kln)), 'kl_consumption':float(np.mean(klg)),
            'kl_portfolio':float(np.mean(kln)), 'value_rrmse':float(rrmse),
            'mean_consumption_relerr':float(np.sqrt(np.mean(((aq/bq-ap/bp)/(ap/bp))**2))),
            'portfolio_mean_rmse':float(np.sqrt(np.mean((mq-mp)**2))),
            'portfolio_var_relerr':float(np.sqrt(np.mean(((vq-vp)/vp)**2)))}

def train(seed=20261125,free_shape=False,steps=3000,batch=1024,h=.01,lrphi=.004,lractor=.003):
    rng=np.random.default_rng(seed);sol=benchmark()
    phi,w=initial_parameters(free_shape)
    op=AdamAscent(len(phi));ow=AdamAscent(len(w));records=[]
    clock=time.perf_counter()
    records.append({'iteration':0,'transitions':0,'seconds':0.,**evaluate(phi,w,free_shape,sol)})
    for k in range(1,steps+1):
        t,y,i,c,p,j,yp,ym,reward=simulator(rng,h,batch)
        cur=value_features(t,y,i)
        nxt=.5*(value_features(t+h,yp,j)+value_features(t+h,ym,j))
        q,grad,*_=policy_eval(w,t,c,p,i,free_shape)
        residual=((nxt-cur)@phi+GT*(.5*(yp+ym)-y)+reward-(q-LAM*y)*h)/h
        gp=np.mean(cur*residual[:,None],axis=0)
        gw=np.mean(grad*residual[:,None],axis=0)
        decay=(1+k/1000)**(-.55)
        op.step(phi,gp,lrphi*decay);ow.step(w,gw,lractor*decay)
        if k%50==0 or k==steps:
            records.append({'iteration':k,'transitions':2*batch*k,'seconds':time.perf_counter()-clock,**evaluate(phi,w,free_shape,sol)})
            if not np.all(np.isfinite(w)):raise RuntimeError('Nonfinite parameters')
    out={'seed':seed,'free_shape':free_shape,'steps':steps,'batch':batch,'h':h,
         'lrphi':lrphi,'lractor':lractor,'phi':phi.tolist(),'w':w.tolist(),'records':records}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--batch',type=int,default=1024)
    ap.add_argument('--seed',type=int,default=20261125);ap.add_argument('--free-shape',action='store_true');ap.add_argument('--h',type=float,default=.01)
    ap.add_argument('--lrphi',type=float,default=.004);ap.add_argument('--lractor',type=float,default=.003)
    a=ap.parse_args();out=train(a.seed,a.free_shape,a.steps,a.batch,a.h,a.lrphi,a.lractor)
    dest=ROOT/('free' if a.free_shape else 'fixed');dest.mkdir(exist_ok=True)
    path=dest/f'seed_{a.seed}_h_{a.h:g}.json';path.write_text(json.dumps(out,indent=2))
    print(path,flush=True);print(json.dumps(out['records'][-1]),flush=True)

if __name__=='__main__':main()
