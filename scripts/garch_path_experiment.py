"""Isolated exploratory experiment; no notebook or baseline edits."""
from pathlib import Path
import hashlib,json,warnings
import numpy as np
import pandas as pd
from arch import arch_model
from distribution_anatomy import french

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/garch_path'
SEED=71931

def simulate(h0,params,date,n=8192,seed=SEED,constant_control=False):
    omega,alpha,beta,gamma,nu=params
    rng=np.random.default_rng(seed+int(date.strftime('%Y%m%d')))
    variance=np.full(n,h0);pnl=np.zeros(n);low=np.zeros(n)
    fixed=np.zeros(n);fixed_low=np.zeros(n)
    for _ in range(20):
        z=rng.standard_t(nu,n//2)*np.sqrt((nu-2)/nu);z=np.concatenate([z,-z])
        ret=np.sqrt(variance)*z;pnl+=ret;low=np.minimum(low,pnl)
        variance=np.maximum(omega+(alpha+gamma*(ret<0))*ret**2+beta*variance,1e-12)
        if constant_control:
            fixed+=np.sqrt(h0)*z;fixed_low=np.minimum(fixed_low,fixed)
    probability=lambda x:float(((x<=-5).sum()+.5)/(n+1))
    result=[probability(low),probability(pnl)]
    assert result[1]<=result[0]
    if constant_control:result += [probability(fixed_low),probability(fixed)]
    assert np.isfinite(result).all()
    return result

def losses(y,p):
    p=np.clip(p,1e-12,1-1e-12)
    return (y-p)**2,-y*np.log(p)-(1-y)*np.log1p(-p)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    files=['research/recent_validation/path.parquet','research/local_stress/barrier_fixed.parquet','data/raw/portfolios.zip']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    z=pd.read_parquet(ROOT/files[0]);old=pd.read_parquet(ROOT/files[1])
    pd.testing.assert_frame_equal(z.loc[old.index,['y','p_fhs','p_vol']],old[['y','p_fhs','p_vol']])
    z=z.loc[z.index.isin(old.index)|(z.index.year>=2023)].copy()
    legs=french('portfolios.zip').loc['1964':]
    daily=(legs['SMALL HiPRIOR']+legs['BIG HiPRIOR']-legs['SMALL LoPRIOR']-legs['BIG LoPRIOR'])/2
    for date,r in z.iterrows():
        j=daily.index.searchsorted(date,side='right');p=daily.iloc[j:j+20].cumsum()
        assert len(p)==20 and p.index[-1]==r.end
        np.testing.assert_allclose([min(0.,p.min()),p.iloc[-1]],[r.path_min,r.terminal],atol=1e-10)
    audits=[];repeat=[]
    for year in sorted(z.index.year.unique()):
        te=z.loc[z.index.year==year];cutoff=te.index.min();past=daily.loc[:cutoff]
        for name,o in [('garch_t',0),('gjr_t',1)]:
            model=arch_model(past,mean='Zero',vol='GARCH',p=1,o=o,q=1,dist='t',rescale=False)
            with warnings.catch_warnings(record=True) as caught:
                fit=model.fit(disp='off',options={'maxiter':1500})
            if fit.convergence_flag!=0:raise RuntimeError(f'Fit failed: {year} {name}')
            p=fit.params;params=tuple(float(p.get(k,0)) for k in ['omega','alpha[1]','beta[1]','gamma[1]','nu'])
            omega,alpha,beta,gamma,nu=params
            assert omega>0 and alpha>=0 and beta>=0 and alpha+gamma>=-1e-8 and nu>2
            r=float(past.iloc[-1]);h=float(fit.conditional_volatility.iloc[-1]**2)
            hnext=max(omega+(alpha+gamma*(r<0))*r*r+beta*h,1e-12)
            np.testing.assert_allclose(hnext,fit.forecast(horizon=1,reindex=False).variance.iloc[-1,0],rtol=1e-8)
            audits.append(dict(year=int(year),model=name,fit_end=str(past.index[-1].date()),n_train=len(past),omega=omega,alpha=alpha,beta=beta,gamma=gamma,nu=nu,persistence=alpha+beta+gamma/2,warnings=[str(w.message) for w in caught],converged=True))
            previous=cutoff
            for date in te.index:
                for ret in daily.loc[(daily.index>previous)&(daily.index<=date)]:
                    hnext=max(omega+(alpha+gamma*(ret<0))*ret**2+beta*hnext,1e-12)
                previous=date
                values=simulate(hnext,params,date,constant_control=o==0)
                z.loc[date,'p_'+name]=values[0];z.loc[date,'endpoint_'+name]=values[1]
                z.loc[date,'initial_variance_'+name]=hnext
                if o==0:
                    z.loc[date,'p_constant_t']=values[2];z.loc[date,'endpoint_constant_t']=values[3]
                    if year>=2023:
                        v=simulate(hnext,params,date,n=32768,seed=SEED+800000,constant_control=True)
                        repeat.append(dict(date=date,p_garch_t=v[0],endpoint_garch_t=v[1],p_constant_t=v[2],endpoint_constant_t=v[3]))
        print(f'Completed {year}',flush=True)
    models=['p_fhs','p_vol','p_garch_t','p_gjr_t','p_constant_t'];scores=[];pairs=[]
    masks={'1984_2022':z.index.year<=2022,'2023_plus':z.index.year>=2023,'all':np.ones(len(z),bool)}
    masks.update({str(y):z.index.year==y for y in sorted(set(z.index.year))})
    for subset,mask in masks.items():
        a=z.loc[mask]
        for model in models:
            b,l=losses(a.y,a[model]);rec=dict(subset=subset,model=model,n=len(a),events=int(a.y.sum()),predicted=float(a[model].mean()),observed=float(a.y.mean()),brier=float(b.mean()),logloss=float(l.mean()))
            if 'endpoint_'+model[2:] in a:
                ep=a['endpoint_'+model[2:]];yb=(a.terminal<=-5).astype(int);eb,el=losses(yb,ep)
                rec.update(endpoint_predicted=float(ep.mean()),endpoint_observed=float(yb.mean()),endpoint_brier=float(eb.mean()),endpoint_logloss=float(el.mean()))
            scores.append(rec)
        if subset.isdigit():continue
        for model,base in [('p_garch_t','p_fhs'),('p_garch_t','p_vol'),('p_garch_t','p_constant_t'),('p_gjr_t','p_fhs'),('p_gjr_t','p_garch_t')]:
            for metric,k in [('brier',0),('logloss',1)]:
                delta=losses(a.y,a[model])[k]-losses(a.y,a[base])[k]
                v=delta.reindex(pd.date_range(a.index.min(),a.index.max(),freq='W-FRI')).to_numpy()
                for block in [13,26,52]:
                    rng=np.random.default_rng(SEED);boot=[]
                    for _ in range(1000):
                        starts=rng.integers(len(v),size=int(np.ceil(len(v)/block)));ix=((starts[:,None]+np.arange(block))%len(v)).ravel()[:len(v)]
                        boot.append(np.nanmean(v[ix]))
                    pairs.append(dict(subset=subset,model=model,baseline=base,metric=metric,block=block,difference=float(np.nanmean(v)),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975))))
    rep=pd.DataFrame(repeat).set_index('date');mc=[]
    for model in ['p_garch_t','p_constant_t']:
        a=z.loc[rep.index];diff=rep[model]-a[model]
        b,l=losses(a.y,rep[model]);mc.append(dict(model=model,mean_absolute_probability_change=float(diff.abs().mean()),max_absolute_probability_change=float(diff.abs().max()),repeat_predicted=float(rep[model].mean()),repeat_brier=float(b.mean()),repeat_logloss=float(l.mean())))
    z.to_parquet(OUT/'predictions.parquet');rep.to_parquet(OUT/'mc_repeat.parquet')
    for name,obj in [('scores',scores),('pairs',pairs),('fits',audits),('mc_check',mc)]:
        (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2))
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (OUT/'QA.json').write_text(json.dumps(dict(status='PASS',source_hashes=hashes,script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_hash=hashlib.sha256((ROOT/'research/GARCH_PATH_PROTOCOL.md').read_bytes()).hexdigest(),fits=len(audits),n=len(z),checks=['All labels replay','Original baseline probabilities replay','All fits converged','One-step variance matches arch','All simulated endpoint probabilities <= path probabilities','Source and original forecast hashes unchanged','Recent Monte Carlo repeat completed']),indent=2))
    print(pd.DataFrame(scores).query("subset in ['1984_2022','2023_plus']").to_string(index=False))
    print(pd.DataFrame(pairs).query("subset=='2023_plus' and metric=='logloss' and block==26").to_string(index=False))
    print(pd.DataFrame(mc).to_string(index=False))

if __name__=='__main__':main()
