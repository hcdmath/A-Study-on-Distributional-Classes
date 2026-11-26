"""Validate, summarize, and plot the reproducible Section 5.3 experiments."""
import json, csv, platform, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp, quad
from scipy.linalg import expm
from scipy.special import gammaln, digamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import (ROOT, T, RHO, LAM, GT, MARKET, gamma_t, exact_A,
                        benchmark, basis, policy_eval, value_features,
                        initial_parameters)


def policy_value(w, free_shape):
    def rhs(t, b):
        ts=np.array([t,t]); ii=np.array([0,1])
        _,_,a,rate,m,v=policy_eval(w,ts,np.ones(2),np.zeros(2),ii,free_shape)
        sig=MARKET['sigma']; th=MARKET['mu']-MARKET['R']
        drift=MARKET['R']+th*m-a/rate-.5*sig**2*(m*m+v)
        hc=a-np.log(rate)+gammaln(a)+(1-a)*digamma(a)
        hp=.5*np.log(2*np.pi*np.e*v)
        source=exact_A(t)*drift+gamma_t(t)*(digamma(a)-np.log(rate))+LAM*(hc+hp)
        return -MARKET['Q']@b-source
    sol=solve_ivp(rhs,[T,0],np.zeros(2),method='DOP853',rtol=2e-11,atol=2e-12,dense_output=True)
    assert sol.success
    return sol


def population_grid(h):
    K=round(T/h); tt=np.arange(K+1)*h
    aa=np.zeros(K+1);bb=np.zeros((K+1,2));aa[-1]=GT
    ph=expm(MARKET['Q']*h)
    r=MARKET['R'];sig=MARKET['sigma'];th=MARKET['mu']-r
    for k in range(K-1,-1,-1):
        g=gamma_t(tt[k]);alpha=1+g/LAM;a=aa[k+1]
        aa[k]=a+(g+LAM)*h
        f=a*(r+th**2/(2*sig**2))+LAM*gammaln(alpha)+(g+LAM)*np.log(LAM/a)+LAM/2*np.log(2*np.pi*LAM/(a*sig**2))
        bb[k]=ph@bb[k+1]+h*f
    return tt,aa,bb


def validate(sol):
    def v(t,x,i):return exact_A(t)*np.log(x)+sol.sol(t)[i]
    max_hjb=0.
    for t in [.03,.21,.57,.93]:
        for x in [.5,1.,2.]:
            for i in [0,1]:
                dt=1e-5;dx=x*2e-4
                vt=(v(t+dt,x,i)-v(t-dt,x,i))/(2*dt)
                p=(v(t,x+dx,i)-v(t,x-dx,i))/(2*dx)
                z=(v(t,x+dx,i)-2*v(t,x,i)+v(t,x-dx,i))/dx**2
                g=gamma_t(t);a=1+g/LAM;r=MARKET['R'][i];sig=MARKET['sigma'][i];th=MARKET['mu'][i]-r
                h=r*x*p-th**2*p*p/(2*sig**2*z)+LAM*gammaln(a)+(g+LAM)*np.log(LAM/p)+LAM/2*np.log(2*np.pi*LAM/(-sig**2*x*x*z))
                residual=vt+sum(MARKET['Q'][i,j]*v(t,x,j) for j in [0,1])+h
                max_hjb=max(max_hjb,abs(residual))
    norms=[]
    for t in [0.,.5,1.]:
        a=1+gamma_t(t)/LAM;rate=exact_A(t)/LAM
        nc=quad(lambda c:np.exp(a*np.log(rate)-gammaln(a)+(a-1)*np.log(c)-rate*c),0,np.inf,epsabs=1e-11)[0]
        norms.append(abs(nc-1))
        for i in [0,1]:
            m=(MARKET['mu'][i]-MARKET['R'][i])/MARKET['sigma'][i]**2
            var=LAM/(exact_A(t)*MARKET['sigma'][i]**2)
            nw=quad(lambda w:np.exp(-(w-m)**2/(2*var))/np.sqrt(2*np.pi*var),-np.inf,np.inf,epsabs=1e-11)[0]
            norms.append(abs(nw-1))
    graderr=0.
    for free in [False,True]:
        _,w=initial_parameters(free)
        w+=np.random.default_rng(102).normal(0,.04,len(w))
        t=np.array([.1,.4,.8]);c=np.array([.2,1.,1.7]);p=np.array([-.5,.3,1.2]);i=np.array([0,1,0])
        _,grad,*_=policy_eval(w,t,c,p,i,free)
        for j in range(len(w)):
            wp=w.copy();wm=w.copy();wp[j]+=1e-6;wm[j]-=1e-6
            fd=(policy_eval(wp,t,c,p,i,free)[0]-policy_eval(wm,t,c,p,i,free)[0])/2e-6
            graderr=max(graderr,np.max(abs(fd-grad[:,j])))
    assert max_hjb<1e-5 and max(norms)<1e-8 and graderr<1e-6
    return {'finite_difference_hjb_max_residual':max_hjb,'density_normalization_max_error':max(norms),'actor_gradient_max_error':graderr}


def coords(x,y):
    return ' '.join('({:.8g},{:.8g})'.format(a,b) for a,b in zip(x,y))


def main():
    sol=benchmark(); checks=validate(sol)
    allruns={}
    for name in ['fixed','free']:
        allruns[name]=[json.loads(p.read_text()) for p in sorted((ROOT/name).glob('seed_*_h_0.01.json'))]
        assert len(allruns[name])==10
    summary={'validation':checks,'benchmark':{'A0':float(exact_A(0)), 'B0':sol.sol(0).tolist()},'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},'methods':{}}
    rows=[]
    for name,runs in allruns.items():
        outputs=[]
        for run in runs:
            kl=np.array([r['kl'] for r in run['records']]);it=np.array([r['iteration'] for r in run['records']])
            hits=[j for j in range(len(kl)-2) if np.all(kl[j:j+3]<.01)]
            hit=int(it[hits[0]]) if hits else None
            pv=policy_value(np.array(run['w']),run['free_shape'])
            gaps=sol.sol(0)-pv.sol(0)
            assert np.min(gaps)>-1e-7
            row={'method':name,'seed':run['seed'],**run['records'][-1], 'hitting_iteration':hit,'utility_gap':float(np.mean(gaps)),'gap_regime_1':float(gaps[0]),'gap_regime_2':float(gaps[1])}
            rows.append(row);outputs.append(row)
        keys=['kl','value_rrmse','mean_consumption_relerr','portfolio_mean_rmse','portfolio_var_relerr','seconds','hitting_iteration','utility_gap']
        summary['methods'][name]={key:{'mean':float(np.mean([r[key] for r in outputs])),'sd':float(np.std([r[key] for r in outputs],ddof=1)),'median':float(np.median([r[key] for r in outputs]))} for key in keys}
        summary['methods'][name]['mean_B_policy_0']=np.mean([policy_value(np.array(r['w']),r['free_shape']).sol(0) for r in runs],axis=0).tolist()
    with (ROOT/'run_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    mesh=[]
    for h in [.04,.02,.01,.005,.0025]:
        tt,aa,bb=population_grid(h);err=0.
        for y in [-1.,1.]:
            err=max(err,float(np.max(abs(aa[:,None]*y+bb-exact_A(tt)[:,None]*y-sol.sol(tt).T))))
        mesh.append({'h':h,'max_value_error':err,'A0_error':abs(float(aa[0]-exact_A(0))),'B0_max_error':float(np.max(abs(bb[0]-sol.sol(0))))})
    for k,row in enumerate(mesh):row['order']=None if k==0 else float(np.log2(mesh[k-1]['max_value_error']/row['max_value_error']))
    summary['mesh']=mesh
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (ROOT/'mesh.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(mesh[0]));writer.writeheader();writer.writerows(mesh)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    colors={'fixed':'#1768AC','free':'#CC6600'};labels={'fixed':'Fixed known shape','free':'Learned shape'}
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    pgf=[r'\begin{figure}[tbp]',r'\centering',r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.25cm},width=0.45\textwidth,height=5.1cm,xlabel={Updates},ymode=log,grid=major,legend style={font=\scriptsize,at={(0.98,0.98)},anchor=north east},tick label style={font=\scriptsize},label style={font=\small}]']
    for panel,(key,ylabel,texlabel) in enumerate([('kl','Mean policy KL',r'$E_{\mathrm{KL}}$'),('value_rrmse','Value relative RMSE',r'$E_V$')]):
        ax=axs[panel];pgf.append(r'\nextgroupplot[ylabel={'+texlabel+'}]')
        for name,runs in allruns.items():
            its=np.array([r['iteration'] for r in runs[0]['records']]);vals=np.array([[r[key] for r in run['records']] for run in runs]);mean=vals.mean(0);sd=vals.std(0,ddof=1)
            ax.semilogy(its,mean,color=colors[name],label=labels[name]);ax.fill_between(its,np.maximum(mean-sd,1e-7),mean+sd,color=colors[name],alpha=.16)
            # Every second recorded checkpoint, retaining the final checkpoint.
            ix=np.arange(0,len(its),2);co='blue!70!black' if name=='fixed' else 'orange!85!black'
            pgf.extend([r'\addplot[name path='+name+'upper,draw=none,forget plot] coordinates {'+coords(its[ix],(mean+sd)[ix])+'};',r'\addplot[name path='+name+'lower,draw=none,forget plot] coordinates {'+coords(its[ix],np.maximum(mean-sd,1e-7)[ix])+'};',r'\addplot['+co+r',fill opacity=0.12,forget plot] fill between[of='+name+'upper and '+name+'lower];',r'\addplot['+co+r',thick] coordinates {'+coords(its[ix],mean[ix])+'};',r'\addlegendentry{'+labels[name]+'}'])
        ax.set_xlabel('Updates');ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    pgf.extend([r'\end{groupplot}',r'\end{tikzpicture}',r'\caption{Learning curves over ten paired seeds. Solid curves show the arithmetic mean; shaded bands show one sample standard deviation. Each update uses 1,024 antithetic pairs, or 2,048 observed transitions.}',r'\label{fig:learning-curves}',r'\end{figure}'])
    (ROOT/'learning_figure.tex').write_text('\n'.join(pgf))
    fig.savefig(ROOT/'learning_curves.pdf');fig.savefig(ROOT/'learning_curves.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained');tt=np.linspace(0,T,101)
    pgf=[r'\begin{figure}[tbp]',r'\centering',r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.25cm},width=0.45\textwidth,height=5cm,xlabel={$t$},grid=major,legend style={font=\scriptsize,at={(0.02,0.98)},anchor=north west},tick label style={font=\scriptsize},label style={font=\small}]',r'\nextgroupplot[ylabel={Mean consumption / wealth}]']
    truth=(gamma_t(tt)+LAM)/exact_A(tt)
    vals=[]
    for r in allruns['fixed']:
        _,_,a,b,_,_=policy_eval(np.array(r['w']),tt,np.ones_like(tt),np.zeros_like(tt),np.zeros(len(tt),int),False);vals.append(a/b)
    learned=np.mean(vals,axis=0)
    axs[0].plot(tt,truth,'k-',label='Benchmark');axs[0].plot(tt,learned,'--',color=colors['fixed'],label='Learned, seed mean')
    ix=np.arange(0,101,5)
    pgf.extend([r'\addplot[black,thick] coordinates {'+coords(tt[ix],truth[ix])+'};',r'\addlegendentry{Benchmark}',r'\addplot[blue!70!black,thick,dashed] coordinates {'+coords(tt[ix],learned[ix])+'};',r'\addlegendentry{Learned, seed mean}',r'\nextgroupplot[ylabel={Portfolio standard deviation}]'])
    for i in [0,1]:
        truth=np.sqrt(LAM/(exact_A(tt)*MARKET['sigma'][i]**2));vals=[]
        for r in allruns['fixed']:
            *_,v=policy_eval(np.array(r['w']),tt,np.ones_like(tt),np.zeros_like(tt),np.full(len(tt),i),False);vals.append(np.sqrt(v))
        learned=np.mean(vals,axis=0);co=['blue!70!black','orange!85!black'][i];mc=[colors['fixed'],colors['free']][i]
        axs[1].plot(tt,truth,'-',color=mc,label=f'Benchmark, regime {i+1}');axs[1].plot(tt,learned,'--',color=mc,label=f'Learned, regime {i+1}')
        pgf.extend([r'\addplot['+co+r',thick] coordinates {'+coords(tt[ix],truth[ix])+'};',r'\addlegendentry{Benchmark, regime '+str(i+1)+'}',r'\addplot['+co+r',thick,dashed] coordinates {'+coords(tt[ix],learned[ix])+'};',r'\addlegendentry{Learned, regime '+str(i+1)+'}'])
    for ax in axs:ax.set_xlabel('Time');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[0].set_ylabel('Mean consumption / wealth');axs[1].set_ylabel('Portfolio standard deviation')
    pgf.extend([r'\end{groupplot}',r'\end{tikzpicture}',r'\caption{Policy moments for the logarithmic benchmark and the fixed-shape learner after 5,000 updates, averaged over ten seeds. Consumption divided by wealth is independent of the current regime in this particular benchmark.}',r'\label{fig:policy-moments}',r'\end{figure}'])
    (ROOT/'policy_figure.tex').write_text('\n'.join(pgf))
    fig.savefig(ROOT/'policy_moments.pdf');fig.savefig(ROOT/'policy_moments.png');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
