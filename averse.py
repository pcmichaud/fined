import numpy as np
import pandas as pd
from scipy.optimize import brentq, root_scalar, minimize
from functools import partial
from scipy.stats import norm
import numdifftools as nd
from scipy.integrate import quad
import multiprocessing as mp
import warnings
warnings.simplefilter(action='ignore')

# load data and cleanup
df = pd.read_stata('clean_May2023.dta',convert_categoricals=False)

df['dmu'] = df['mu_post'] - df['mu_pre']
df['dsig'] = df['sig_post'] - df['sig_pre']


df['treat_offer'] = df['treat_offer'].astype('int64')
df['treat'] = np.where(df['treat'].isna(),0,df['treat'])
for v in ['mu_pre','mu_post','sig_pre','sig_post']:
    df[v] *= 1e-2

df['lottery'] = np.where(df['q13_1']==0,1,0)
for i in range(2,11):
    df['lottery'] = np.where((df['q13_'+str(i)]==0) & (df['q13_'+str(i-1)]==1),i,df['lottery'])
df['lottery'] = np.where(df['lottery']==0,10,df['lottery'])

# payoff table and derivation of bounds
payA, payB = [20,16], [39,1]
pr = np.arange(0.1,1.1,0.1)

def crra(wealth,sigma):
    if sigma==1.0:
        return np.log(wealth)
    else :
        return (wealth**(1-sigma)-1.0)/(1.0-sigma)

def eu(wealth,p,pay,sigma):
    return p*crra(wealth+pay[0],sigma) + (1-p)*crra(wealth+pay[1],sigma)

def q(sigma,wealth,p,payA,payB):
    return eu(wealth,p,payA,sigma) - eu(wealth,p,payB,sigma)

def solve(wealth,p,payA,payB):
    sigma = brentq(q,-2.0,2.0,args=(wealth,p,payA,payB))
    return sigma

cuts = [solve(0.0,p,payA,payB) for p in pr[:-1]]
cuts.insert(0,-2.0)
cuts.append(3.0)

js = range(1,11)
bounds = pd.DataFrame(data=None,index=None,columns=['lottery','sigma_min','sigma_max'])
bounds['lottery'] = js
bounds['sigma_min'] = cuts[:-1]
bounds['sigma_max'] = cuts[1:]

# merge back bounds
df = df.merge(bounds,on='lottery')

# estimation of distribution of risk aversion
Phi = norm(0,1).cdf
phi = norm(0,1).pdf
def loglike(par,x,sigma_max,sigma_min):
    m = x.shape[1]
    beta = par[:m]
    eta = np.exp(par[m])
    xb = np.matmul(x,beta)
    bmax = (sigma_max - xb)/eta
    bmin = (sigma_min - xb)/eta
    p = Phi(bmax) - Phi(bmin)
    p = np.where(p<1e-10,1e-10,p)
    lli = np.log(p)
    return -np.sum(lli)
df['_const'] = 1.0
df['agep'] = (df['age']-df['age'].mean())/df['age'].std()
df['log_finwlth'] = np.log(1.0+df['finwlth'])
xnames = ['agep','female','married',
          'educ_somecollege','educ_college','anykids','log_hhincome','hhincome_miss','log_finwlth','_const']

x = df[xnames].to_numpy()
k = x.shape[1]
ipar = [0.0 for i in range(k)]
ipar.append(np.log(0.25))
ipar = np.array(ipar)
opt = minimize(loglike,ipar,args=(x,df['sigma_max'].values,df['sigma_min'].values),method='BFGS')

pnames = ['age (z)','female','married',
          'some college','college','any kids','log hh income','income missing','log fin. wealth','constant']
pnames.append('$\\eta$')

H = nd.Hessian(loglike)(opt.x,x,df['sigma_max'].values,df['sigma_min'].values)
invH = np.linalg.inv(H)
se = np.sqrt(np.diag(invH))

beta = opt.x[:k]
eta = np.exp(opt.x[k])
se[k] = eta*se[k]

df['mu_sigma'] = np.matmul(x,beta)
df['eta_sigma'] = eta
results = pd.DataFrame(data=np.append(beta,eta),index=pnames,columns=['point estimate'])
results['standard error'] = se
results = results.round(3)

# computation of CE
def itgr_u(eps,w0,sigma,mu,sig):
    return crra(max(w0*(1+mu+sig*eps),0.001),sigma)*phi(eps)

# define EU
w0 = 30
nreps_eps = 500
np.random.seed(1234)
eps = np.random.normal(size=(nreps_eps))
eps = np.where(eps>3,3,eps)
eps = np.where(eps<-3,-3,eps)
def eu(w0,sigma,mu,sig):
    #result = np.mean([crra(max(w0*(1+mu+sig*e),1e-6),sigma) for e in eps])
    result = quad(itgr_u,-3,3,args=(w0,sigma,mu,sig),epsabs=1e-3,epsrel=1.e-3)
    return result[0]

def solve(ce,w0,sigma,mus,sigs):
    eu_pre = eu(w0,sigma,mus[0],sigs[0])
    eu_post = eu(w0*(1-ce),sigma,mus[1],sigs[1])
    return eu_pre - eu_post

def ce(w0,sigma,mus,sigs):
    a = -1
    b = 0.999
    solve_a = solve(a,w0,sigma,mus,sigs)
    solve_b = solve(b,w0,sigma,mus,sigs)
    if np.sign(solve_a)!=np.sign(solve_b):
        value = brentq(solve,a,b,args=(w0,sigma,mus,sigs))
    else :
        if np.abs(solve_a) < np.abs(solve_b):
            value = a
        else :
            value = b
    return value
#exit()

def wgt(sigma,sigmas,mu_s,eta_s):
    Phi_max = Phi((sigmas[1] - mu_s)/eta_s)
    Phi_min = Phi((sigmas[0] - mu_s)/eta_s)
    if Phi_max != Phi_min:
        wgt = (1/eta_s)*phi((sigma - mu_s)/eta_s)/(Phi_max - Phi_min)
    else :
        wgt = 0.0
    return wgt

def ce_row(row):
    mus = [row['mu_pre'], row['mu_post']]
    sigs = [row['sig_pre'],row['sig_post']]
    sigmas = [row['sigma_min'],row['sigma_max']]
    mu_s = row['mu_sigma']
    eta_s = row['eta_sigma']
    span = np.linspace(sigmas[0],sigmas[1],50)
    wgts = [wgt(sigma,sigmas,mu_s,eta_s) for sigma in span]
    ces = [ce(w0,sigma,mus,sigs) for sigma in span]
    totwgt = np.sum(wgts)
    if totwgt>0:
        wgts = [w/totwgt for w in wgts]
        row['gain_ce'] =np.sum([(c>0)*w for c,w in zip(ces,wgts)])
        row['mean_ce'] = np.sum([c*w for c,w in zip(ces,wgts)])
    else :
        row['gain_ce'] = 0.0
        row['mean_ce'] = 0.0
    return row

df['gain_ce'] = np.nan
df['mean_ce'] = np.nan
def process_partition(partition):
    partition = partition.apply(ce_row,axis=1)
    return partition

if __name__ == '__main__':
    npartitions = 16
    df = df.sample(n=500)
    list_df = np.array_split(df, npartitions)
    with mp.Pool(processes=npartitions) as pool:
        res = pool.map(process_partition, list_df)
        df = pd.concat(res)
    print(df.groupby(['treat_offer','treat']).mean()[['gain_ce']].to_latex())
    df.to_stata('Clean_March2024.dta')


